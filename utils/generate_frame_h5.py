import os
import time
import h5py
import numpy as np
import pandas as pd
import tonic
from PIL import Image
from omegaconf import OmegaConf
from src.utils.transformations import Transformations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback


class SamplesDataLoader(tonic.Dataset):
    def __init__(self, dataset_dir, dataset_type="synthetic", transform=None):
        """
        Args:
            dataset_dir (str): Root directory containing the synthetic or Real dataset folders.
            dataset_type (str): Type of dataset to load ("Synthetic" or "Real").
            transform (callable, optional): Optional transform to apply to the events.
        """
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.samples = self._load_samples()
        self.transform = transform

    def _load_samples(self):
        """Load event and label files and pair them appropriately."""
        samples = []
        events_dir = os.path.join(self.dataset_dir, "h5")
        if not os.path.isdir(events_dir):
            print(f"Directory not found: {events_dir}")
            return samples
        for file_name in sorted(os.listdir(events_dir)):
            if file_name.endswith(".h5") and file_name[0].isalpha():
                event_file = os.path.join(events_dir, file_name)
                samples.append(event_file)
        return samples

    def _load_h5_data(self, h5_path):
        """Loads events and labels from a given HDF5 file."""
        try:
            with h5py.File(h5_path, 'r') as f:
                event_grp = f['events']
                # More efficient creation of the structured array
                events = np.empty(len(event_grp['ts']), dtype=[('t', '<f8'), ('x', '<i4'), ('y', '<i4'), ('p', '<i4')])
                events['t'] = event_grp['ts'][:]
                events['x'] = event_grp['xs'][:]
                events['y'] = event_grp['ys'][:]
                events['p'] = event_grp['ps'][:]

                labels = f['labels']['data']
                labels_df = pd.DataFrame({
                    'filename': labels['filename'].astype(str),
                    't_xyz': list(zip(labels['Tx'], labels['Ty'], labels['Tz'])),
                    'q_xyzw': list(zip(labels['Qx'], labels['Qy'], labels['Qz'], labels['Qw'])),
                    't': labels['timestamp']
                })
            return events, labels_df
        except Exception as e:
            print(f"Error loading H5 file {h5_path}: {e}")
            return None, None

    def _frames_from_label_intervals(self, events, labels_df, label_time_scale=1.0, include_last=True):
        """For every timestamp t_i in labels_df['t'], collect events in [t_i, t_{i+1})."""
        if events is None or len(events) == 0:
            return [], pd.DataFrame()

        evt_t = events['t']
        if not np.all(np.diff(evt_t) >= 0):
            events = np.sort(events, order='t')
            evt_t = events['t']

        labels_df = labels_df.sort_values('t').reset_index(drop=True)

        # Gracefully handle cases with too few labels to form an interval
        if len(labels_df) < 2:
            return [], pd.DataFrame()

        lbl_times = labels_df['t'].to_numpy(dtype=float) * label_time_scale
        idxs = np.searchsorted(evt_t, lbl_times, side='left')

        frames, meta_data = [], []

        # Iterate up to the second to last label to form intervals [t_i, t_{i+1})
        for i in range(len(lbl_times) - 1):
            start_idx, end_idx = idxs[i], idxs[i + 1]
            slice_events = events[start_idx:end_idx]

            frame = self.transform(events=slice_events) if self.transform else slice_events
            frames.append(frame)

            meta = labels_df.iloc[i].to_dict()
            meta.update({
                't_start': lbl_times[i],
                't_end': lbl_times[i + 1],
                'num_events': len(slice_events)
            })
            meta_data.append(meta)

        # Handle the last interval from the last label to the end of events
        if include_last:
            start_idx = idxs[-1]
            slice_events = events[start_idx:]

            frame = self.transform(events=slice_events) if self.transform else slice_events
            frames.append(frame)

            meta = labels_df.iloc[-1].to_dict()
            meta.update({
                't_start': lbl_times[-1],
                't_end': evt_t[-1] if len(evt_t) > 0 else lbl_times[-1],
                'num_events': len(slice_events)
            })
            meta_data.append(meta)

        return frames, pd.DataFrame(meta_data)

    def __getitem__(self, idx):
        """Load and return a sample from the dataset at the given index."""
        event_file = self.samples[idx]
        events, labels_df = self._load_h5_data(event_file)

        if events is None or len(events) == 0 or labels_df is None or labels_df.empty:
            return None, None

        evt_range = events['t'].max() - events['t'].min()
        lbl_range = labels_df['t'].max() - labels_df['t'].min()
        label_time_scale = 1e-6 if evt_range > 0 and lbl_range / evt_range > 1000 else 1.0

        frames, interval_labels = self._frames_from_label_intervals(
            events, labels_df, label_time_scale=label_time_scale, include_last=True
        )

        return frames, interval_labels

    def __len__(self):
        return len(self.samples)


def process_and_save_sample(args):
    """Worker function to process and save a single sample. Designed to be robust."""
    idx, sample_path, transformation_name, output_dir, cfg_dict = args

    try:
        # Re-create config from dict for the transformation instance
        cfg = OmegaConf.create(cfg_dict)
        transformation_instance = Transformations(cfg)

        event_representations = {
            "two_polarity_time_surface": transformation_instance.two_polarity_time_surface,
            "two_d_histogram": transformation_instance.two_d_histogram,
            "lnes": transformation_instance.lnes,
            "to_voxel_grid": transformation_instance.to_voxel_grid,
            "three_c_representation": transformation_instance.three_c_representation,
            "event_frame": transformation_instance.event_frame,
        }
        transform = event_representations.get(transformation_name)
        if not transform:
            return f"❌ Sample {idx} failed: Unknown transformation '{transformation_name}'"

        # The data loader is now minimal and only loads a single specified file
        loader = SamplesDataLoader(os.path.dirname(os.path.dirname(sample_path)), transform=transform)
        loader.samples = [sample_path]  # Process only the one sample assigned to this worker

        event_frames, _ = loader[0]
        traj_name = os.path.basename(sample_path).split('.')[0]

        if not event_frames:
            return f"- Sample {idx} ({traj_name}) skipped (no frames generated)."

        sample_output_dir = os.path.join(output_dir, traj_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        for i, frame in enumerate(event_frames):
            # Normalize and convert frame for image saving
            if frame.dtype != np.uint8:
                if frame.max() > 0:
                    frame = (frame / frame.max() * 255)
                frame = frame.astype(np.uint8)

            if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:  # (C, H, W)
                frame = frame.transpose(1, 2, 0)

            im = Image.fromarray(frame)
            im.save(os.path.join(sample_output_dir, f"img{i:04d}_{traj_name}.png"))

        return f"✅ Sample {idx} ({traj_name}) saved successfully."
    except Exception:
        # Catch ANY exception, print traceback, and return an error message
        tb_str = traceback.format_exc()
        return f"❌ Sample {idx} ({os.path.basename(sample_path)}) CRASHED. Traceback:\n{tb_str}"


if __name__ == "__main__":
    try:
        cfg = OmegaConf.load("configs/mobilenet_heatmap.yaml")
    except Exception as e:
        print(f"FATAL: Error loading YAML configuration: {e}")
        exit()

    dataset_dir = cfg.root.dataset
    output_dir = os.path.join(cfg.root.data_out, cfg.data.transformation)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}\n")

    # Create a dummy loader just to get the list of samples
    initial_loader = SamplesDataLoader(dataset_dir=dataset_dir, dataset_type=cfg.data.source)
    all_samples = initial_loader.samples
    num_samples = len(all_samples)

    if num_samples == 0:
        print("No samples found. Exiting.")
        exit()

    print(f"Event representation: {cfg.data.transformation}")
    print(f"Dataset type: {cfg.data.source}\n")

    # Pass config as a dict, which is safer for multiprocessing
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args_list = [(idx, sample_path, cfg.data.transformation, output_dir, cfg_dict)
                 for idx, sample_path in enumerate(all_samples)]

    start_time = time.time()
    num_workers = max(1, int(cpu_count() * 0.5))
    print(f"Starting processing of {num_samples} samples with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered with tqdm for a responsive progress bar
        results = list(tqdm(pool.imap_unordered(process_and_save_sample, args_list), total=num_samples))

    print("\n--- Processing Summary ---")
    for msg in sorted(results):
        print(msg)

    successful_saves = sum(1 for r in results if r.startswith('✅'))
    print(f"\nDONE -- {successful_saves}/{num_samples} samples processed successfully.")
    print("Time taken: {:.2f} seconds".format(time.time() - start_time))