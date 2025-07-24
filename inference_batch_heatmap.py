import ast  # For safely parsing the bbox string (e.g., "[x0, y0, x1, y1]")
import os

import tensorflow as tf
import tensorflow.keras as keras
from omegaconf import OmegaConf

from src.dataloaders.spades import create_dataset
from src.models.mobilenet_heatmap import mobilenet_heatmap_small
from src.utils.kdsnt import spatial_expectation2d


def convert_to_original_size(coordst, gt_df, csv_path_out):
    gt_df['bbox'] = gt_df['bbox'].apply(ast.literal_eval)
    bboxes = np.array(gt_df['bbox'])
    print(bboxes)
    x0 = gt_df['bbox'].apply(lambda b: b[0]).to_numpy().reshape(-1, 1)
    y0 = gt_df['bbox'].apply(lambda b: b[1]).to_numpy().reshape(-1, 1)
    x1 = gt_df['bbox'].apply(lambda b: b[2]).to_numpy().reshape(-1, 1)
    y1 = gt_df['bbox'].apply(lambda b: b[3]).to_numpy().reshape(-1, 1)
    coords = coordst.numpy()
    # Compute width and height (with small epsilon to avoid division by zero)
    eps = 1e-6
    widths = x1 - x0 + eps  # shape: (35940, 1)
    heights = y1 - y0 + eps

    # Recover relative positions from normalized coords
    rel_coords = 0.5 * (coords + 1.0)
    rel_coords[:, :, 0] *= widths  # scale x
    rel_coords[:, :, 1] *= heights  # scale y

    # Add bbox top-left to get absolute positions
    rel_coords[:, :, 0] += x0  # x + x0
    rel_coords[:, :, 1] += y0  # y + y0

    # coords: shape (35940, 8, 2)
    x_coords = rel_coords[:, :, 0]  # shape (35940, 8)
    y_coords = rel_coords[:, :, 1]  # shape (35940, 8)

    # Concatenate x and y along axis 1 to get shape (35940, 16)
    coords_flat = np.concatenate([x_coords, y_coords], axis=1)
    x_cols = [f'x{i}' for i in range(8)]
    y_cols = [f'y{i}' for i in range(8)]
    columns = x_cols + y_cols

    # Create DataFrame
    coords_df = pd.DataFrame(coords_flat, columns=columns)

    # This regex will remove the full prefix including the RT-folder (like RT001/, RT023/, etc.)
    gt_df['filepath'] = gt_df['filepath'].str.replace(r'^/home/arunkumar/datasets/SPADES/synthetic/lnes_cropped'
                                                      r'/test/', '', regex=True)
    filepaths = gt_df['filepath'].values
    # Add filepaths column
    coords_df['filepath'] = filepaths

    # Move 'filepath' to the front
    coords_df = coords_df[['filepath'] + columns]

    coords_df.to_csv(csv_path_out, index=False)


def convert_relative_kpts_to_absolute(input_csv, output_csv):
    # Read the CSV
    df = pd.read_csv(input_csv)
    # Parse bbox strings into actual lists
    df['bbox'] = df['bbox'].apply(ast.literal_eval)
    # Extract bbox top-left corner
    df['bbox_x0'] = df['bbox'].apply(lambda b: b[0])
    df['bbox_y0'] = df['bbox'].apply(lambda b: b[1])
    # Compute absolute keypoints by adding bbox origin
    for i in range(8):
        df[f'x{i}'] = df[f'k{i}x'] + df['bbox_x0']
        df[f'y{i}'] = df[f'k{i}y'] + df['bbox_y0']
    df['filepath'] = df['filepath'].str.replace(r'^/home/arunkumar/datasets/SPADES/synthetic/lnes_cropped'
                                                r'/test/', '', regex=True)
    # Prepare final columns
    x_cols = [f'x{i}' for i in range(8)]
    y_cols = [f'y{i}' for i in range(8)]
    final_columns = ['filepath'] + x_cols + y_cols
    # Create final DataFrame and save
    df_out = df[final_columns]
    df_out.to_csv(output_csv, index=False)


import pandas as pd
import numpy as np


def compute_mean_keypoint_error(pred_csv, gt_csv, num_keypoints=8):
    """
    Compute mean keypoint error (Euclidean distance) between prediction and ground truth CSVs.

    Args:
        pred_csv (str): Path to predictions CSV.
        gt_csv (str): Path to ground truth CSV.
        num_keypoints (int): Number of keypoints per sample (default 8).

    Returns:
        float: Mean keypoint error across all keypoints and samples.
    """
    # Load both CSVs
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    # Keep only rows that are common by 'filepath'
    merged_df = pd.merge(pred_df, gt_df, on='filepath', suffixes=('_pred', '_gt'))

    if merged_df.empty:
        raise ValueError("No matching filepaths found between prediction and ground truth CSVs.")

    # Extract keypoint columns
    pred_kpts = []
    gt_kpts = []
    for i in range(num_keypoints):
        pred_x = merged_df[f'x{i}_pred'].to_numpy()
        pred_y = merged_df[f'y{i}_pred'].to_numpy()
        gt_x = merged_df[f'x{i}_gt'].to_numpy()
        gt_y = merged_df[f'y{i}_gt'].to_numpy()

        pred_kpts.append(np.stack([pred_x, pred_y], axis=1))  # shape: (N, 2)
        gt_kpts.append(np.stack([gt_x, gt_y], axis=1))

    # Stack all keypoints: list of (N, 2) → (N, K, 2)
    pred_kpts = np.stack(pred_kpts, axis=1)  # shape: (N, K, 2)
    gt_kpts = np.stack(gt_kpts, axis=1)  # shape: (N, K, 2)

    # Compute Euclidean distance per keypoint
    errors = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)  # shape: (N, K)

    # Mean error across all keypoints and samples
    mean_error = np.mean(errors)

    return mean_error


if __name__ == '__main__':

    config_path = "configs/mobilenet_heatmap.yaml"
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    # dataset creation
    csv_path = os.path.join(config.root.data_out, 'lnes_cropped/test/keypoints.csv')
    test_df = pd.read_csv(csv_path)
    test_data = {col: test_df[col].values for col in test_df.columns}

    test_dataset = create_dataset(test_data,
                                  batch_size=config.training.batch_size,
                                  input_size=config.data.input_size[:2],
                                  is_training=False,
                                  cache_dir=None)

    # tf.keras.utils.get_custom_objects().update({'PoseEstimationLoss': PoseEstimationLoss})
    input_image_size = config.data.input_size
    input_shape = list(config.data.input_size)  # Convert ListConfig to a standard list
    model = mobilenet_heatmap_small(input_size=list(config.training.input_size), num_keypoints=8)
    model.build(input_shape=(None, *input_shape))  # None is for batch size
    keras.models.load_model(config.model.trained_weights)
    # model.load_weights(config.model.trained_weights)
    predictions = model.predict(test_dataset, verbose=1)

    tf.print(tf.shape(predictions))

    # --- Reshaping logic (remains the same and is crucial) ---
    # if tf.rank(predictions) == 4:
    #     pred_shape = tf.shape(predictions)
    #     # if pred_shape[1] > pred_shape[3]:
    #     y_pred_logits = tf.transpose(predictions, perm=[0, 3, 1, 2])
    # elif tf.rank(predictions) == 3:
    #     y_pred_logits = tf.expand_dims(predictions, axis=1)

    # Convert the model's logits into a probability heatmap.
    # predicted_heatmap = dsnt.flat_softmax(y_pred_logits)
    # n_coords = dsnt.soft_argmax(predicted_heatmap)
    predictions = tf.convert_to_tensor(predictions)  # ensures tf.Tensor
    # spatial_softmax = spatial_softmax2d(predictions)
    n_coords = spatial_expectation2d(predictions)

    pred_csv_path = "./tmp/test_predictions.csv"
    convert_to_original_size(n_coords, test_df, pred_csv_path)

    gt_csv = "./tmp/test_ground_truth.csv"
    convert_relative_kpts_to_absolute(csv_path, gt_csv)

    mean_error = compute_mean_keypoint_error(pred_csv_path, gt_csv, num_keypoints=8)
    print(mean_error)
