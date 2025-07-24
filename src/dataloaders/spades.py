import os
from typing import Tuple

import tensorflow as tf

from src.utils.kdsnt import normalize_pixel_coordinates


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
        heatmap: Boolean indicating the desired output format for positions.
        cache_dir: Optional directory to cache preprocessed data. If None, no caching is used.

    Returns:
        A TensorFlow Dataset object.
    """

    def _load_and_preprocess(filepath: tf.Tensor) -> tf.Tensor:
        """Loads, preprocesses, and converts a single image to a tensor."""
        # filepath = tf.strings.regex_replace(
        #     filepath,
        #     pattern="^/home/arunkumar",  # Match /home at the start of the string
        #     rewrite="/data/arathinam"
        # )
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)  # Or decode_png, depending on your image format
        # image = tf.image.resize(image, input_size)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        return image

    def _parse_bbox(bbox_str):
        """
        Parses a string like '[x0, y0, x1, y1]' into a float tensor.
        """
        s = tf.strings.regex_replace(bbox_str, r'\[|\]| ', '')
        parts = tf.strings.split(s, sep=',')
        return tf.strings.to_number(parts, out_type=tf.float32)

    def _process_data(item):
        """Process each item: load image and stack positions/quaternions"""
        image = _load_and_preprocess(item["filepath"])
        rel_bbox_positions = []
        for i in range(8):
            px = tf.cast(item[f'k{i}x'], tf.float32)
            py = tf.cast(item[f'k{i}y'], tf.float32)
            rel_bbox_positions.append([px, py])
        rel_positions = tf.stack(rel_bbox_positions, axis=0)  # Shape: [8, 2]
        bbox = _parse_bbox(item['bbox'])
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        normalized_positions = normalize_pixel_coordinates(rel_positions, height=bbox_height, width=bbox_width)

        if heatmap:
            output_positions = normalized_positions
        else:
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
