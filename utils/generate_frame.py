import os
import shutil
import time
import sys
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import multiprocessing as mp
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tonic
from PIL import Image
from omegaconf import OmegaConf

from src.utils.transformations import Transformations


class SamplesDataLoader(tonic.Dataset):
    def __init__(self, dataset_dir: str, dataset_type: str = "synthetic", 
                 transform: Optional[callable] = None, transform_name: Optional[str] = None,
                 num_workers: int = 0):
        """
        Args:
            dataset_dir (str): Root directory containing the synthetic or Real dataset folders.
            dataset_type (str): Type of dataset to load ("synthetic" or "Real").
            transform (callable, optional): Optional transform to apply to the events.
            transform_name (str, optional): Name of the transformation to apply.
            num_workers (int): Number of worker processes for parallel processing (0 = no parallelization).
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_type = dataset_type
        self.transform = transform
        self.transform_name = transform_name
        self.num_workers = num_workers
        
        # Cache file paths to avoid repeated filesystem operations
        self.samples = self._load_samples()
        
        # Pre-allocate dtype for faster CSV loading
        self._event_dtype = np.dtype([('t', 'float64'), ('x', 'int32'), ('y', 'int32'), ('p', 'int32')])

    def _load_samples(self) -> List[Tuple[str, Optional[str]]]:
        """
        Load event and label files and pair them appropriately.

        Returns: list of tuples: Each tuple contains paths to an event file and its corresponding label file (for
        synthetic data).
        """
        samples = []
        events_dir = self.dataset_dir / self.dataset_type / "events"

        if not events_dir.exists():
            raise FileNotFoundError(f"Events directory not found: {events_dir}")

        # Use list comprehension for better performance
        csv_files = [f for f in events_dir.iterdir() 
                    if f.suffix == ".csv" and f.name[0].isalpha()]

        if self.dataset_type == "synthetic":
            labels_dir = self.dataset_dir / self.dataset_type / "labels"
            if not labels_dir.exists():
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
            
            samples = [(str(event_file), str(labels_dir / event_file.name)) 
                      for event_file in csv_files]
        elif self.dataset_type == "Real":
            samples = [(str(event_file), None) for event_file in csv_files]

        print(f"Loaded {len(samples)} samples")
        return samples

    @lru_cache(maxsize=128)
    def _load_events(self, file_path: str) -> np.ndarray:
        """
        Load event data from a CSV file and return it as a structured ndarray with specific data types.
        Uses caching to avoid reloading the same files.

        Args:
            file_path (str): Path to the event CSV file.

        Returns:
            np.ndarray: Structured array of events with fields ['t', 'x', 'y', 'p'].
        """
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"Loading events from {os.path.basename(file_path)} ({file_size:.2f} MB)")
            
            load_start = time.time()
            
            # Optimized CSV loading with explicit dtype and better parameters
            events_df = pd.read_csv(
                file_path, 
                header=None, 
                names=['t', 'x', 'y', 'p'],
                dtype={'t': 'float64', 'x': 'int32', 'y': 'int32', 'p': 'int32'},
                on_bad_lines='skip',
                engine='c',  # Use C engine for better performance
                memory_map=True  # Memory map for large files
            )

            # Convert directly to structured array without intermediate list comprehension
            events = events_df.to_records(index=False)
            events = events.astype(self._event_dtype)

            load_time = time.time() - load_start
            
            # Validate the loaded data
            if len(events) == 0:
                raise ValueError(f"No valid events found in {file_path}")

            print(f"Loaded {len(events):,} events in {load_time:.3f}s")
            return events

        except Exception as e:
            print(f"Error loading events from {file_path}: {e}")
            raise

    @lru_cache(maxsize=128)
    def _load_labels(self, file_path: str) -> np.ndarray:
        """
        Load label data from a CSV file with caching.

        Args:
            file_path (str): Path to the label CSV file.
            
        Returns:
            np.ndarray: Structured array of labels.
        """
        try:
            labels = pd.read_csv(file_path, engine='c')
            return labels.to_records(index=False)
        except Exception as e:
            print(f"Error loading labels from {file_path}: {e}")
            raise

    def __getitem__(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and return a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Tuple containing the event frames and labels (if available).
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
            
        event_file, label_file = self.samples[idx]
        traj_name = Path(event_file).stem
        
        print(f"Processing sample {idx}: {traj_name}")
        sample_start = time.time()
        
        try:
            events = self._load_events(event_file)
            
            if len(events) == 0:
                print(f"Unable to load events for sample at index {idx}")
                return None, None

            labels = None
            if label_file is not None:
                labels = self._load_labels(label_file)

            # Transform events and create frames
            transform_start = time.time()
            
            if self.transform is not None:
                event_frames = self.transform(events=events)
                if len(event_frames) != 600:
                    print(f"Warning: {self.transform_name} generated {len(event_frames)} frames instead of 600")
            else:
                event_frames = events
            
            transform_time = time.time() - transform_start
            sample_time = time.time() - sample_start
            
            if isinstance(event_frames, (list, np.ndarray)):
                frame_count = len(event_frames) if hasattr(event_frames, '__len__') else 1
                print(f"Generated {frame_count} frames in {transform_time:.3f}s")
            else:
                print(f"Transformation completed in {transform_time:.3f}s")

            return event_frames, labels
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return None, None

    def save_sample(self, idx: int, output_dir: str) -> bool:
        """
        Save a sample from the dataset at the given index to a file.

        Args:
            idx (int): Index of the sample to save.
            output_dir (str): Directory where the sample should be saved.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get the trajectory name from the file path
            event_file_path = Path(self.samples[idx][0])
            traj_name = event_file_path.stem

            print(f"Saving sample {idx}: {traj_name}")
            save_start = time.time()

            event_frames, labels = self.__getitem__(idx)

            if event_frames is None:
                print(f"No frames to save for sample {idx}")
                return False

            # Create output directory
            sample_dir = Path(output_dir) / traj_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save frames efficiently
            saved_count = self._save_frames(event_frames, sample_dir, traj_name)
            
            save_time = time.time() - save_start
            print(f"Saved {saved_count} frames in {save_time:.3f}s")

            return True
            
        except Exception as e:
            print(f"Error saving sample {idx}: {e}")
            return False

    def _save_frames(self, event_frames: np.ndarray, output_dir: Path, traj_name: str) -> int:
        """
        Efficiently save event frames as images.
        
        Args:
            event_frames: Array of frames to save
            output_dir: Directory to save frames to
            traj_name: Name of the trajectory for file naming
            
        Returns:
            int: Number of frames successfully saved
        """
        # Determine if we need to transpose frames
        transpose_needed = self.transform_name == "lnes"
        saved_count = 0
        
        print(f"Saving {len(event_frames)} frames...")
        
        for i, frame in enumerate(event_frames):
            try:
                # Prepare frame for saving
                if transpose_needed and len(frame.shape) == 3:
                    frame_to_save = frame.transpose(1, 0, 2)
                else:
                    frame_to_save = frame
                
                # Ensure frame is uint8
                if frame_to_save.dtype != np.uint8:
                    frame_to_save = np.clip(frame_to_save, 0, 255).astype(np.uint8)
                
                # Create and save image
                im = Image.fromarray(frame_to_save)
                filename = f"img{i:03d}_{traj_name}.png"
                im.save(output_dir / filename)
                saved_count += 1

                if i % 100 == 0:
                    print(f"Saved {i} frames")
                
            except Exception as e:
                print(f"Error saving frame {i} for trajectory {traj_name}: {e}")
                continue
        
        return saved_count


def process_sample_worker(args):
    """Worker function for parallel processing of samples."""
    data_loader, idx, output_dir = args
    worker_id = mp.current_process().name
    print(f"Worker {worker_id} starting sample {idx}")
    
    try:
        start_time = time.time()
        success = data_loader.save_sample(idx, output_dir)
        elapsed = time.time() - start_time
        
        if success:
            print(f"Worker {worker_id} completed sample {idx} in {elapsed:.2f}s")
        else:
            print(f"Worker {worker_id} failed sample {idx} in {elapsed:.2f}s")
            
        return idx, success, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Worker {worker_id} error on sample {idx} after {elapsed:.2f}s: {e}")
        return idx, False, str(e)


if __name__ == "__main__":
    """
    Optimized data generation script with improved performance and error handling.
    Configurations for the data loader can be changed at configs/mobilenet_heatmap.yaml
    """

    print("Starting Data Generation Pipeline")
    print("=" * 60)
    
    # Load configuration file
    print("Loading configuration...")
    try:
        cfg = OmegaConf.load("configs/mobilenet_heatmap.yaml")
        print("Configuration loaded successfully")
        print(f"   Dataset: {cfg.root.dataset}")
        print(f"   Transformation: {cfg.data.transformation}")
        print(f"   Dataset type: {cfg.data.source}")
        print(f"   Input size: {cfg.data.input_size}")
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)

    # Setup directories
    dataset_dir = cfg.root.dataset
    output_dir = Path(cfg.root.data_out) / cfg.data.transformation
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # Check if dataset directory exists
    if not Path(dataset_dir).exists():
        print(f"Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    # Load configuration parameters
    print(f"\nInitializing transformation: {cfg.data.transformation}")
    transformation_instance = Transformations(cfg)

    # Mapping of event representations
    event_representations = {
        "two_polarity_time_surface": transformation_instance.two_polarity_time_surface,
        "two_d_histogram": transformation_instance.two_d_histogram,
        "lnes": transformation_instance.lnes,
        "to_voxel_grid": transformation_instance.to_voxel_grid,
        "three_c_representation": transformation_instance.three_c_representation,
        "event_frame": transformation_instance.event_frame,
    }

    # Initialize data loader with optimized settings
    transform_func = event_representations[cfg.data.transformation]
    dataset_type = cfg.data.source
    
    # Determine number of workers (0 = no parallelization)
    num_workers = min(4, mp.cpu_count())  # Limit to 4 workers to avoid memory issues
    
    print(f"Transformation initialized")

    data_loader = SamplesDataLoader(
        dataset_dir=dataset_dir, 
        dataset_type=dataset_type, 
        transform=transform_func, 
        transform_name=cfg.data.transformation,
        num_workers=num_workers
    )
    print(f"Data loader initialized with {len(data_loader.samples)} samples")

    # Process samples
    start_time = time.time()
    
    # Define which trajectories to process (can be modified as needed)
    trajs_to_process = [200]
    # Filter indices to only process specified trajectories
    indices_to_process = [idx for idx in range(len(data_loader.samples)) if idx in trajs_to_process]
    
    print(f"\nProcessing {len(indices_to_process)} samples from {len(data_loader.samples)} total samples")
    print("=" * 60)

    if num_workers > 0 and len(indices_to_process) > 1:
        # Parallel processing
        print("Starting parallel processing...")
        parallel_start = time.time()
        
        with mp.Pool(processes=num_workers) as pool:
            args_list = [(data_loader, idx, str(output_dir)) for idx in indices_to_process]
            results = pool.map(process_sample_worker, args_list)
        
        parallel_time = time.time() - parallel_start
        print(f"\nParallel processing completed in {parallel_time:.2f}s")
        
        # Process results
        successful = 0
        failed = 0
        total_processing_time = 0
        
        for idx, success, error in results:
            if success:
                print(f"Sample {idx} completed successfully")
                successful += 1
            else:
                print(f"Sample {idx} failed: {error}")
                failed += 1
        
    else:
        # Sequential processing
        print("Starting sequential processing...")
        successful = 0
        failed = 0
        
        for i, idx in enumerate(indices_to_process):
            sample_start = time.time()
            
            if data_loader.save_sample(idx, str(output_dir)):
                sample_time = time.time() - sample_start
                print(f"Sample {idx} completed in {sample_time:.2f}s")
                successful += 1
            else:
                sample_time = time.time() - sample_start
                print(f"Sample {idx} failed in {sample_time:.2f}s")
                failed += 1
            

    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Successful samples: {successful}")
    print(f"Failed samples: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    if successful > 0:
        print(f"Output directory: {output_dir}")
    
    print("=" * 60)
