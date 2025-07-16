import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
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

    def _normalize_keypoints(rel_positions, bbox):
        """
        Normalizes absolute keypoint positions to the [-1, 1] range relative to a bounding box.
        """
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox_width = x1 - x0 + 1e-6
        bbox_height = y1 - y0 + 1e-6
        bbox_dims = tf.stack([bbox_width, bbox_height])
        normalized_positions = 2.0 * (rel_positions / bbox_dims) - 1.0
        return normalized_positions, (rel_positions / bbox_dims) * 224

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
        normalized_positions, resized_input_positions = _normalize_keypoints(rel_positions, bbox)

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


# --- New helper function for visualization ---
def denormalize_points(points, image_shape):
    """
    Converts points from [-1, 1] normalized space to pixel coordinates.
    Args:
        points (np.ndarray): Array of normalized points with shape (num_points, 2).
        image_shape (tuple): The (height, width) of the image.
    Returns:
        np.ndarray: Array of points in pixel coordinates.
    """
    height, width = image_shape[0], image_shape[1]
    # Denormalize from [-1, 1] to [0, 1]
    points_01 = (points + 1.0) / 2.0
    # Scale to image dimensions
    pixel_coords = points_01 * np.array([width, height])
    return pixel_coords


if __name__ == '__main__':
    config_path = "/home/arunkumar/dev-python/2024-nc-hackathon-AstroSpikes/configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    csv_path = os.path.join(cfg.root.data_out, 'lnes_cropped/val/keypoints.csv')
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    # Convert DataFrame to a dictionary of NumPy arrays for tf.data.Dataset
    data_dict = {col: df[col].values for col in df.columns}

    # --- Create a dataset specifically for visualization ---
    # We set is_training=False to prevent shuffling and repeating.
    vis_dataset = create_dataset(data_dict,
                                 batch_size=4,  # Let's visualize 4 images
                                 input_size=cfg.data.input_size[:2],
                                 is_training=False,
                                 heatmap=True,
                                 cache_dir=None)

    # --- Visualization Script ---
    print("\nVisualizing a batch of data to verify correctness...")

    # Get one batch of data
    for images, keypoints_batch in vis_dataset.take(1):
        # Convert tensors to numpy arrays for plotting
        images_np = images.numpy()
        keypoints_np = keypoints_batch.numpy()
        batch_size = images_np.shape[0]
        fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
        if batch_size == 1:  # Ensure axes is always an array
            axes = [axes]

        print(f"Displaying {batch_size} images from the batch...")

        for i in range(batch_size):
            image = images_np[i]
            keypoints_normalized = keypoints_np[i]
            # Denormalize keypoints to plot them on the image
            keypoints_pixel = denormalize_points(keypoints_normalized, image.shape)
            ax = axes[i]
            ax.imshow(image)
            # Plot keypoints as red 'x' markers
            ax.scatter(keypoints_pixel[:, 0], keypoints_pixel[:, 1], c='r', marker='x', s=50)
            ax.set_title(f'Sample {i + 1}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
