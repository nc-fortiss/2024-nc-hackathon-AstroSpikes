import csv
import os
from typing import Tuple

import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


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


def create_dataset(df: pd.DataFrame, batch_size: int, input_size: Tuple[int, int], is_training: bool = True,
                   cache_dir: str = None) -> tf.data.Dataset:
    """
    Creates a TensorFlow Dataset from a Pandas DataFrame for large datasets, optimizing memory usage.

    Args:
        df: Pandas DataFrame containing filepaths and labels.
        batch_size: The batch size for the dataset.
        input_size: A tuple specifying the desired image size (height, width).
        is_training: Boolean indicating whether the dataset is for training. If True, data will be shuffled.
        cache_dir: Optional directory to cache preprocessed data. If None, no caching is used.

    Returns:
        A TensorFlow Dataset object.
    """

    def _load_and_preprocess(filepath: tf.Tensor) -> tf.Tensor:
        """Loads, preprocesses, and converts a single image to a tensor."""
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)  # Or decode_png, depending on your image format
        image = tf.image.resize(image, input_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Normalize to [0, 1]
        return image

    def _prepare_data(filepath: tf.Tensor, Tx: tf.Tensor, Ty: tf.Tensor, Tz: tf.Tensor, Qx: tf.Tensor, Qy: tf.Tensor,
                      Qz: tf.Tensor, Qw: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Loads the image, positions and quaternions and returns a tuple of (image, (position, quaternion)).
        """
        image = _load_and_preprocess(filepath)
        position = tf.stack([Tx, Ty, Tz], axis=-1)
        quaternion = tf.stack([Qx, Qy, Qz, Qw], axis=-1)
        return image, (position, quaternion)

    # Convert DataFrame columns to TensorFlow tensors
    filepaths = tf.constant(df['filepath'].values)
    Tx = tf.constant(df['Tx'].values, dtype=tf.float32)
    Ty = tf.constant(df['Ty'].values, dtype=tf.float32)
    Tz = tf.constant(df['Tz'].values, dtype=tf.float32)
    Qx = tf.constant(df['Qx'].values, dtype=tf.float32)
    Qy = tf.constant(df['Qy'].values, dtype=tf.float32)
    Qz = tf.constant(df['Qz'].values, dtype=tf.float32)
    Qw = tf.constant(df['Qw'].values, dtype=tf.float32)

    # Create a tf.data.Dataset from the tensors
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, Tx, Ty, Tz, Qx, Qy, Qz, Qw))

    if is_training:
        # Create a dataset that repeats indefinitely for training
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=df.shape[0])

    # Map the data preparation function to the dataset
    dataset = dataset.map(_prepare_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache the preprocessed data
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist
        cache_file = os.path.join(cache_dir, "dataset_cache")
        dataset = dataset.cache(cache_file)

    dataset = dataset.batch(batch_size, drop_remainder=True)  # Drop the last incomplete batch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Improves performance by prefetching data

    return dataset
