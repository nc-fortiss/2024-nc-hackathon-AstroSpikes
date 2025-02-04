import os
import tensorflow as tf
import pandas as pd

from omegaconf import OmegaConf
import logging

#from transformations import Transformations
#from filters import Filters

class ImageDataLoader:
    def __init__(self, test=False):
        # Load configuration file
        self.config = OmegaConf.load("conf/global_conf.yaml")

        self.root = self.config.paths.output_dir + '/' + self.config.transformation.method
        self.batch_size = self.config.batch_size
        self.shuffle_count = self.config.shuffle_count
        self.normalize = True
        self.transform = self.center_crop_240x240
        self.test = test
    
    def __call__(self):
        dataset = self.load_dataset_from_directory(self.root)
        train, test = self.split_dataset(dataset)
        train = train.shuffle(self.shuffle_count).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test = test.shuffle(self.shuffle_count).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return train, test

    def split_dataset(self, dataset):
        dataset_size = dataset.cardinality().numpy()
        test_size = int(self.config.test_split * dataset_size)

        test_dataset = dataset.take(test_size)
        train_dataset = dataset.skip(test_size)

        return train_dataset, test_dataset

    def load_image_and_label(self, image_path, label):
        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        if self.normalize:
            image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

         # Apply the transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Convert label to tensor
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        return image, label


    #generate a list of frame paths and corresponding labels
    def load_dataset_from_directory(self, root_directory):
        image_paths = []
        labels = []
        list_dir = os.listdir(root_directory)

        for subdir in list_dir:
            
            subdir_path = os.path.join(root_directory, subdir)
            
            if os.path.isdir(subdir_path):

                label_csv_path = os.path.join(subdir_path, self._get_csv_filename(subdir))
                label_df = pd.read_csv(label_csv_path, header=None, skiprows=1)
                
                for index, row in label_df.iterrows():
                    frame_name = row[0]
                    label = row[1:8].tolist()

                    image_path = os.path.join(subdir_path, frame_name)
                    image_path = os.path.abspath(image_path)
                    
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        labels.append(label)
                    else :
                        print("Picture not found : " + str(image_path))

        image_paths = tf.constant(image_paths)
        labels = tf.constant(labels, dtype=tf.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

# Load the dataset
#root_directory = "/home/brini/hackathon/image_dataset"

#dataset = ImageDataLoader(root_directory, batch_size=32)()

    def _get_csv_filename(self, dirname: str) -> str:
        return dirname + '.csv'

    @staticmethod
    def center_crop_240x240(image):
        """
        Center crops the image to 240x240.
        """
        return tf.image.resize_with_crop_or_pad(image, target_height=224, target_width=224)
