import tonic
import tonic.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import shutil
from functools import partial
import time

from omegaconf import OmegaConf

from transformations import Transformations
from filters import Filters


class SamplesDataLoader(tonic.Dataset):
    def __init__(self, dataset_dir, dataset_type="synthetic", transform=None, filter=None):
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
        self.filter = filter


    def _load_samples(self):
        """
        Load event and label files and pair them appropriately.
        
        Returns:
            list of tuples: Each tuple contains paths to an event file and its corresponding label file (for synthetic data).
        """
        samples = []
        events_dir = os.path.join(self.dataset_dir, self.dataset_type, "events")
        
        if self.dataset_type == "synthetic":
            labels_dir = os.path.join(self.dataset_dir, self.dataset_type, "labels")
            for file_name in sorted(os.listdir(events_dir)):

                if file_name.endswith(".csv") and file_name[0].isalpha():
                    event_file = os.path.join(events_dir, file_name)
                    label_file = os.path.join(labels_dir, file_name)
                    samples.append((event_file, label_file))
        elif self.dataset_type == "Real":
            for file_name in sorted(os.listdir(events_dir)):
                if file_name.endswith(".csv") and file_name[0].isalpha():
                    event_file = os.path.join(events_dir, file_name)
                    samples.append((event_file, None))
        return samples

    def _load_events(self, file_path):
        """
        Load event data from a CSV file and return it as a structured ndarray with specific data types.

        Args:
            file_path (str): Path to the event CSV file.

        Returns:
            np.ndarray: Structured array of events with fields ['t', 'x', 'y', 'p'].
        """
        # Define the expected dtype for the structured array
        dtype = [('t', 'float64'), ('x', 'int32'), ('y', 'int32'), ('p', 'int32')]

        # Load CSV data without a header and assign column names directly
        events_df = pd.read_csv(file_path, header=None, names=['t', 'x', 'y', 'p'], on_bad_lines='skip')

        # Convert the DataFrame to a structured array with the specified dtype
        events = np.array([tuple(row) for row in events_df.to_numpy()], dtype=dtype)

        assert isinstance(events, np.ndarray)

        # Check if the array is a structured array
        assert events.dtype.names is not None

        # Check if the array has the correct fields
        assert 't' in events.dtype.names
        assert 'x' in events.dtype.names
        assert 'y' in events.dtype.names
        assert 'p' in events.dtype.names

        # events = self.transform(events)

        return events

    def _load_labels(self, file_path):
        """
        Load label data from a CSV file.
        
        Args:
            file_path (str): Path to the label CSV file.
        """
        labels = pd.read_csv(file_path)
        return labels.to_records(index=False)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to return.
        
        Returns:
            dict: A dictionary containing the event data and its corresponding label (if available).
        """
        event_file, label_file = self.samples[idx]
        events = self._load_events(event_file)

        if len(events) == 0:
            print("Unable to load events for sample at index ", idx)
            return None, None

        if label_file is not None:
            labels = self._load_labels(label_file)
        else:
            labels = None
        
        if self.filter(events=events) is False:
            return (None, None)
        
        #transform events and create frames
        event_frames = self.transform(events=events)

        sample = event_frames, labels
        return sample

    def save_sample(self, idx, file_path) -> bool:
        """
        Save a sample from the dataset at the given index to a file.
        
        Args:
            idx (int): Index of the sample to save.
            file_path (str): Path to the file where the sample should be saved.
        """
        #get the trajectory name from the index
        traj_name = self.samples[idx][0].split('/')[-1].split('.')[0]

        #create a folder if it does not exist
        #join the file path with the trajectory name
        # file_path = file_path + '/' + traj_name + '/'
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)

        event_frames, labels = self.__getitem__(idx)

        if event_frames is None or labels is None:
            return False

        file_path = file_path + '/' + traj_name + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # rgb_frames = self.generate_rgb_from_samples(events)
        for i, frame in enumerate(event_frames):
            im = Image.fromarray(frame)
            number = f"{i:03}"
            im.save(f"{file_path}img{number}_{traj_name[4:]}.png")
            im.save(f"{file_path}img{number}_{traj_name[4:]}.png")
        
        # copy csv file to folder
        csv_file_path = dataset_dir + '/synthetic/labels/' + traj_name + '.csv'
        csv_dest_path = file_path + traj_name + '.csv'
        shutil.copy(csv_file_path, csv_dest_path)

        return True

if __name__ == "__main__":

    """
    DONT CHANGE ANYTHING BELOW THIS LINE!!!
    Configurations for the data loader can be changed at conf/global_conf.yaml
    """

    # Load configuration file
    # config = OmegaConf.load("conf/global_conf.yaml")
    try:
        config = OmegaConf.load("conf/global_conf.yaml")
        print(config)
    except Exception as e:
        print("Error loading YAML:", e)

    # Load root directory and output directory
    dataset_dir = config.paths.dataset_dir
    output_dir = config.paths.output_dir + '/' + config.transformation.method
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load configuration parameters
    transformation_instance = Transformations()
    filter_instance = Filters(config.filter.parameters.conditions)

    # Mapping of event representations and filters
    event_representations = {
        "two_polarity_time_surface": transformation_instance.two_polarity_time_surface,
        "two_d_histogram": transformation_instance.two_d_histogram,
        "lnes": transformation_instance.lnes,
        "to_voxel_grid": transformation_instance.to_voxel_grid,
        "three_c_representation": transformation_instance.three_c_representation,
    }

    filters = {
        "get_distribution": filter_instance.get_distribution,
    }

    # Initialize data loader
    t = event_representations[config.transformation.method]
    f = filters[config.filter.method]
    d_type = config.dataset_type

    print(f"Event representation: {config.transformation.method}")
    print(f"Filter: {config.filter.method}")
    print(f"Dataset type: {d_type}\n")

    data_loader = SamplesDataLoader(dataset_dir=dataset_dir, dataset_type=d_type, transform=t, filter=f)
    print(f"Successfully initialized data loader.")

    # Generate RGB frames from samples and save them
    start = time.time()
    print(f"Total samples to process: {len(data_loader.samples)}\n")

    for idx in range(len(data_loader.samples)):
        if data_loader.save_sample(idx, output_dir):
            print(f"✅ Sample {idx + 1}/{len(data_loader.samples)} saved successfully.")
        else:
            print(f"❌ Sample {idx + 1}/{len(data_loader.samples)} skipped (filter or empty).")

    print("\nDONE -- Time taken: {:.2f} seconds".format(time.time() - start))