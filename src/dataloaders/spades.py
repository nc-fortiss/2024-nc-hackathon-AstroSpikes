import os
from typing import Tuple

import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf


def create_dataset(data: dict, batch_size: int, input_size: Tuple[int, int],
                   is_training: bool = True,
                   cache_dir: str = None) -> tf.data.Dataset:
    """
    Creates a TensorFlow Dataset from a Pandas DataFrame for large datasets, optimizing memory usage.

    Args:
        data: dict containing filepaths and labels.
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

    def _process_data(item):
        """Process each item: load image and stack positions/quaternions"""
        image = _load_and_preprocess(item["filepath"])
        position = tf.stack([item["Tx"], item["Ty"], item["Tz"]], axis=-1)
        quaternion = tf.stack([item["Qx"], item["Qy"], item["Qz"], item["Qw"]], axis=-1)
        # # Stack position and quaternion into a single tensor
        # labels = tf.concat([position, quaternion], axis=-1)
        # labels = tf.stack([item["Tx"], item["Ty"], item["Tz"], item["Qx"], item["Qy"], item["Qz"], item["Qw"]], axis=-1)
        return image, (position, quaternion)

    # Create a tf.data.Dataset from the filepaths and individual label tensors
    dataset = tf.data.Dataset.from_tensor_slices(data)

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=len(data[next(iter(data))]))

    dataset = dataset.map(_process_data, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "dataset_cache")
        dataset = dataset.cache(cache_file)

    if is_training:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':

    config_path = "/home/arunkumar/dev-python/2024-nc-hackathon-AstroSpikes/configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)
    train_df = pd.read_csv(os.path.join(cfg.root.data_out, 'train.csv'))
    # Convert DataFrame to a dictionary of NumPy arrays
    train_data = {col: train_df[col].values for col in train_df.columns}

    train_dataset = create_dataset(train_data,
                                   batch_size=cfg.training.batch_size,
                                   input_size=cfg.data.input_size[:2],
                                   is_training=True,
                                   cache_dir=None)

    print("Slices of the training dataset:")
    for features in train_dataset.take(5):  # Takes the first 5 elements for demonstration
        print(features)

    # print(train_dataset.cardinality().numpy())
