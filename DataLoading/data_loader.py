import tonic
import tonic.transforms as transforms
import pandas as pd
import os
import numpy as np

root_dir = "/Users/jost/Downloads/SPADES"

class SamplesDataLoader(tonic.Dataset):
    def __init__(self, root_dir, dataset_type="synthetic", transform=None):
        """
        Args:
            root_dir (str): Root directory containing the synthetic or Real dataset folders.
            dataset_type (str): Type of dataset to load ("Synthetic" or "Real").
            transform (callable, optional): Optional transform to apply to the events.
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.samples = self._load_samples()
        self.transform = transform

    def _load_samples(self):
        """
        Load event and label files and pair them appropriately.
        
        Returns:
            list of tuples: Each tuple contains paths to an event file and its corresponding label file (for synthetic data).
        """
        samples = []
        events_dir = os.path.join(self.root_dir, self.dataset_type, "events")
        
        if self.dataset_type == "synthetic":
            labels_dir = os.path.join(self.root_dir, self.dataset_type, "labels")
            for file_name in sorted(os.listdir(events_dir)):
                if file_name.endswith(".csv"):
                    event_file = os.path.join(events_dir, file_name)
                    label_file = os.path.join(labels_dir, file_name)
                    samples.append((event_file, label_file))
        elif self.dataset_type == "Real":
            for file_name in sorted(os.listdir(events_dir)):
                if file_name.endswith(".csv"):
                    event_file = os.path.join(events_dir, file_name)
                    samples.append((event_file, None))
        return samples

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
        if label_file is not None:
            labels = self._load_labels(label_file)
        else:
            labels = None
        sample = events, labels
        return sample

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
        events_df = pd.read_csv(file_path, header=None, names=['t', 'x', 'y', 'p'])

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

        events = self.transform(events)

        return events

    def _load_labels(self, file_path):
        """
        Load label data from a CSV file.
        
        Args:
            file_path (str): Path to the label CSV file.
        """
        labels = pd.read_csv(file_path)
        return labels.to_records(index=False)
