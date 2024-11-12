import tonic
import tonic.transforms as transforms
from tonic.transforms import ToFrame, Compose
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import numpy as np

path = "/Users/jost/Downloads/SPADES"

class TrajectoryDataSet(tonic.dataset):
    def __init__(self, root_dir, dataset_type="synthetic"):
        """
        Args:
            root_dir (str): Root directory containing the synthetic or Real dataset folders.
            dataset_type (str): Type of dataset to load ("Synthetic" or "Real").
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.samples = self._load_samples()

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

class FrameDataSet(tonic.dataset):
    def __init__(self, event_file, label_file, dataset_type="synthetic"):
        """
        Args:
            event_file (str): Path to the event CSV file.
            label_file (str): Path to the label CSV file.
            dataset_type (str): Type of dataset to load ("Synthetic" or "Real").
        """
        self.event_file = event_file  
        self.label_file = label_file 
        self.dataset_type = dataset_type 

    def _load_frames(self):
        #for every thenth second, accumulate the number of events in that tenth second
        #load the event file
        events = pd.read_csv(self.event_file)
        #decide between synthetic and real data
        if self.dataset_type == "synthetic":
            #create a list of frames
            frames = []
            for i in range(0, len(events), 1000):
                #for each pixel of the frame, add the positive events and subtract the negative events during the time interval
                frame = np.zeros(1280, 720, 3)
                for j in range(i, i+1000):

                    #if true add one, otherwise subtract one
                    if events[j].polarity == 1:
                        frame += events[j].p
                    else:
                        frame -= events[j].p
                frames.append(frame)
        else:
            #create a list of frames
            frames = []
            for i in range(0, len(events), 100000):
                #for each pixel of the frame, add the positive events and subtract the negative events during the time interval
                frame = np.zeros(1280, 720)
                for j in range(i, i+100000):
                    if events[j].polarity == 1:
                        frame += events[j].p
                    else:
                        frame -= events[j].p
                frames.append(frame)

        return frames

    def __load_labels(self):
        labels = []
        labels = pd.read_csv(self.label_file) 
        return labels

    