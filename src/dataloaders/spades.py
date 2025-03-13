import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


# from tensorflow.image import random_flip_left_right, random_brightness


class createDF:
    def __init__(self, cfg: DictConfig):
        self.data_root = str(os.path.join(cfg.root.dataset, cfg.data.source, cfg.data.transformation))
        self.label_root = str(os.path.join(cfg.root.dataset, cfg.data.source, 'labels'))
        all_files = os.listdir(self.label_root)
        df_list = [pd.read_csv(os.path.join(self.label_root, label_file)) for label_file in all_files]
        self.merged_df = pd.concat(df_list, ignore_index=True)
        self.merged_df = self.add_filepath_column(self.merged_df)
        self.merged_df.sample(frac=1).reset_index(drop=True)
        self.batch_size = cfg.training.batch_size

    def add_filepath_column(self, df):
        df['filepath'] = df['filename'].apply(
            lambda x: os.path.join(self.data_root, x.split('_')[1].replace('.png', ''), x))
        return df

    def __call__(self):
        # Split the dataset into train and test
        train_df, val_df = train_test_split(self.merged_df, test_size=0.2, random_state=2025)
        # Ensure train size is divisible by batch_size
        train_size = len(train_df)
        val_size = len(val_df)
        train_df = train_df.iloc[:(train_size // self.batch_size) * self.batch_size]
        # Ensure validation size is divisible by batch_size
        val_df = val_df.iloc[:(val_size // self.batch_size) * self.batch_size]
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
        batch_labels = self.df[['Tx', 'Ty', 'Tz', 'Qx', 'Qy', 'Qz', 'Qw']].iloc[
                       index * self.batch_size: (index + 1) * self.batch_size].values
        # Load images corresponding to the file paths
        batch_images = np.array([self._load_image(filepath) for filepath in batch_filepaths])

        # Convert batch labels to TensorFlow tensors with dtype=tf.float32
        batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        return batch_images, batch_labels

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
