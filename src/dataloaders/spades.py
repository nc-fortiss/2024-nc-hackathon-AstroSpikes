import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import csv


class CreateDF:
    def __init__(self, cfg: DictConfig):
        self.data_root = str(os.path.join(cfg.root.dataset, cfg.data.source, cfg.data.transformation))
        self.label_root = str(os.path.join(cfg.root.dataset, cfg.data.source, 'labels'))
        all_files = os.listdir(self.label_root)
        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=2025)
        self.train_files = train_files
        self.val_files = val_files

    def save_file(self, filepath):
        with open(os.path.join(filepath, 'train_files.csv'), "w", newline="") as file:
            writer = csv.writer(file)
            for item in self.train_files:
                writer.writerow([item])

        with open(os.path.join(filepath, 'val_files.csv'), "w", newline="") as file:
            writer = csv.writer(file)
            for item in self.val_files:
                writer.writerow([item])

    def create_df(self, files):
        df_list = [pd.read_csv(os.path.join(self.label_root, label_file)) for label_file in files]
        df = pd.concat(df_list, ignore_index=True)
        df['filepath'] = df['filename'].apply(
            lambda x: os.path.join(self.data_root, x.split('_')[1].replace('.png', ''), x))
        df.sample(frac=1).reset_index(drop=True)
        return df

    def __call__(self):
        # Split the dataset into train and test
        train_df = self.create_df(self.train_files)
        val_df = self.create_df(self.val_files)
        return train_df, val_df


class ImageDataLoader(tf.keras.utils.Sequence):
    def __init__(self, cfg: DictConfig, df: pd.DataFrame):
        self.config = cfg
        self.batch_size = cfg.training.batch_size
        self.input_image_size = cfg.data.input_size
        self.is_training = True
        self.df = df

    def __getitem__(self, index):
        """Generates a batch of data."""
        # Get the batch of file paths
        batch_filepaths = self.df['filepath'].iloc[index * self.batch_size: (index + 1) * self.batch_size].values
        # Get the batch of labels
        batch_positions = self.df[['Tx', 'Ty', 'Tz']].iloc[
                       index * self.batch_size: (index + 1) * self.batch_size].values
        batch_quaternions = self.df[['Qx', 'Qy', 'Qz', 'Qw']].iloc[
                       index * self.batch_size: (index + 1) * self.batch_size].values
        # Load images corresponding to the file paths
        batch_images = np.array([self._load_image(filepath) for filepath in batch_filepaths])

        # Convert batch labels to TensorFlow tensors with dtype=tf.float32
        batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        batch_positions = tf.convert_to_tensor(batch_positions, dtype=tf.float32)
        batch_quaternions = tf.convert_to_tensor(batch_quaternions, dtype=tf.float32)

        return batch_images, (batch_positions, batch_quaternions)

        # return batch_images, batch_positions

    def _load_image(self, filepath):
        """Loads and preprocesses a single image."""
        image = load_img(filepath, target_size=self.input_image_size)
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]

        return image

    def on_epoch_end(self):
        """Shuffles data at epoch end."""
        return self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        """Number of batches per epoch."""
        return self.df.shape[0] // self.batch_size
