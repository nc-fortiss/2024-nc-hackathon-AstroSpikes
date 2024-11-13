import tonic
import tonic.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import shutil

root_dir = "/Users/jost/Downloads/SPADES"

class SamplesDataLoader(tonic.Dataset):
    def __init__(self, root_dir, dataset_type="synthetic", transform=None, threshold=0.8):
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
        self.threshold = threshold
        self.boundries = [(280, False), (1000, True), (1280, False)]


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
        
        if self._get_distribution(events) < self.threshold:
            return (None, None)
        
        events = self.transform(events)

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


    def _get_distribution(self, events, boundries: list[tuple[int, bool]]=None) -> bool:
        """Filters out the traces where events are outside the center of the image
        boundries: for eg. [(280, False), (360, True)] means that the x values should be between 280 and 360"""
        if boundries is None:
            boundries = self.boundries
            
        band_edges = np.array([boundry[0] for boundry in boundries])
        band_active = np.array([boundry[1] for boundry in boundries])

        print(events[0].shape)

        sample_events_x = np.array([smpl[1] for smpl in events])
        # only three bins
        events_x_digitize = np.digitize(sample_events_x, band_edges, right=True)
        values, counts = np.unique(events_x_digitize, return_counts=True)

        fraction_active = np.sum(counts[band_active]) / np.sum(counts)

        print("Result", fraction_active)
        
        return fraction_active


    def save_sample(self, idx, file_path):
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

        events, labels = self.__getitem__(idx)

        if events is None:
            print(f"Event with index {idx} rejected while filtering.")
            return

        file_path = file_path + '/' + traj_name + '/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        rgb_frames = self.generate_rgb_from_samples(events)
        for i, frame in enumerate(rgb_frames):
            im = Image.fromarray(frame)
            number = f"{i:03}"
            print(number)
            im.save(f"{file_path}img{number}_{traj_name[4:]}.png")
            im.save(f"{file_path}img{number}_{traj_name[4:]}.png")
        
        # copy csv file to folder
        csv_file_path = root_dir + '/synthetic/labels/' + traj_name + '.csv'
        csv_dest_path = file_path + traj_name + '.csv'
        shutil.copy(csv_file_path, csv_dest_path)




    def generate_rgb_from_samples(self, sample_events):
        ret = []
        for frame in range(0, (len(sample_events)//3)*3, 3):
            #empty frame
            rgb_frame = np.zeros((256,256,3), dtype=np.uint8)
            #stack 3 frames into 3 channels
            #scale [0,1] to [0,255]
            rgb_frame[:,:,0] = sample_events[frame]*255
            rgb_frame[:,:,1] = sample_events[frame+1]*255
            rgb_frame[:,:,2] = sample_events[frame+2]*255
            ret.append(rgb_frame)
        return ret
    

if __name__ == "__main__":
        root_dir= "/home/lecomte/AstroSpikes/SPADES"
        output_dir= "./train_dataset" #os.path.join(root_dir, "train_dataset")
        os.makedirs(output_dir, exist_ok=True)
        transform_queue = transforms.Compose([
                    transforms.MergePolarities(),
                    transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                    transforms.Downsample(spatial_factor=256/720),
                    transforms.ToTimesurface(dt=333,tau=200,sensor_size=(256,256,1))
                ])
        dataset = SamplesDataLoader(root_dir=root_dir, dataset_type="synthetic", transform=transform_queue)
        for idx in range(len(dataset.samples)):
            dataset.save_sample(idx, output_dir)
            print(f"Sample {idx} saved successfully.")
             
        print ("DONE")