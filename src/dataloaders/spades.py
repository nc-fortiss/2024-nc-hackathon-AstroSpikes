import os
from typing import Tuple
import json

import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf


def create_dataset(data: dict, batch_size: int, input_size: Tuple[int, int],
                   is_training: bool = True,
                   heatmap: bool = True,
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
        # image = tf.image.resize(image, input_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Normalize to [0, 1]
        return image

    def _parse_bbox(bbox_str):
        """
        Parses a string like '[x0, y0, x1, y1]' into a float tensor.
        """
        # Remove brackets and spaces
        s = tf.strings.regex_replace(bbox_str, r'\[|\]| ', '')
        # Split by comma
        parts = tf.strings.split(s, sep=',')
        # Convert to numbers
        return tf.strings.to_number(parts, out_type=tf.float32)

    def _normalize_keypoints(abs_positions, bbox):
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

        """Normalize keypoints to [0, 1]"""
        bbox_width = x1 - x0 + 1e-6
        bbox_height = y1 - y0 + 1e-6

        bbox_origin = tf.stack([x0, y0])
        bbox_dims = tf.stack([bbox_width, bbox_height])

        # Apply the normalization formula using broadcasting: (pos - origin) / dims
        normalized_positions = (abs_positions - bbox_origin) / bbox_dims

        return normalized_positions

    def _process_data(item):
        """Process each item: load image and stack positions/quaternions"""
        image = _load_and_preprocess(item["filepath"])
        positions = []
        for i in range(8):
            # Ensure keypoints are float32
            px = tf.cast(item[f'k{i}x'], tf.float32)
            py = tf.cast(item[f'k{i}y'], tf.float32)
            positions.append([px, py])
        abs_positions = tf.stack(positions, axis=0)  # Shape: [8, 2]

        bbox = _parse_bbox(item['bbox'])
        normalized_positions = _normalize_keypoints(abs_positions, bbox)
        # 6. Reshape the output based on the 'heatmap' flag
        if heatmap:
            # The shape [8, 2] is ideal for heatmap targets
            output_positions = normalized_positions
        else:
            # For direct regression, flatten to shape [16]
            output_positions = tf.reshape(normalized_positions, [16])

        return image, output_positions

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

    config_path = "/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)
    print(os.path.join(cfg.root.data_out, 'lnes_cropped/val/keypoints.csv'))
    train_df = pd.read_csv(os.path.join(cfg.root.data_out, 'lnes_cropped/val/keypoints.csv'))
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
