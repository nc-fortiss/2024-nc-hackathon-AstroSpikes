import argparse
import json
import warnings
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# --- Constants for SPEED Score Calculation ---
ROTATION_THRESHOLD_RAD = np.deg2rad(0.169)
TRANSLATION_THRESHOLD_NORM = 0.002173
MIN_PNP_POINTS = 4


def calculate_translation_error(
        t_pred: np.ndarray, t_gt: np.ndarray, eps: float = 1e-8
) -> Tuple[float, float]:
    """Calculates absolute and normalized translation error.

    Args:
        t_pred: Predicted translation vector (3,).
        t_gt: Ground truth translation vector (3,).
        eps: A small epsilon to prevent division by zero.

    Returns:
        A tuple containing:
        - error_abs (float): The absolute L2 norm of the translation error.
        - error_norm (float): The error normalized by the ground truth magnitude.
    """
    t_pred = np.asarray(t_pred).reshape(3)
    t_gt = np.asarray(t_gt).reshape(3)

    error_abs = np.linalg.norm(t_gt - t_pred)
    error_norm = error_abs / (np.linalg.norm(t_gt) + eps)

    return float(error_abs), float(error_norm)


def calculate_orientation_error(
        q_pred: np.ndarray, q_gt: np.ndarray
) -> Tuple[float, float]:
    """Calculates the geodesic distance between two quaternions.

    Args:
        q_pred: Predicted quaternion (x, y, z, w).
        q_gt: Ground truth quaternion (x, y, z, w).

    Returns:
        A tuple containing:
        - error_deg (float): The angular error in degrees.
        - error_rad (float): The angular error in radians.
    """
    q_pred = np.asarray(q_pred).reshape(4)
    q_gt = np.asarray(q_gt).reshape(4)

    # Normalize quaternions to ensure they are unit quaternions
    q_pred /= np.linalg.norm(q_pred)
    q_gt /= np.linalg.norm(q_gt)

    # q and -q represent the same rotation, so we use the absolute dot product
    dot_product = np.abs(np.dot(q_pred, q_gt))
    # Clip to handle floating-point inaccuracies
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # The angle between two unit quaternions is 2 * arccos(|q1 . q2|)
    error_rad = 2 * np.arccos(dot_product)
    error_deg = np.degrees(error_rad)

    return float(error_deg), float(error_rad)


def calculate_speed_score(
        t_pred: np.ndarray,
        q_pred: np.ndarray,
        t_gt: np.ndarray,
        q_gt: np.ndarray,
        apply_thresh: bool = True,
) -> Tuple[float, float, float, float, float]:
    """Calculates the SPEED score and other related pose metrics.

    Args:
        t_pred: Predicted translation vector.
        q_pred: Predicted orientation quaternion.
        t_gt: Ground truth translation vector.
        q_gt: Ground truth orientation quaternion.
        apply_thresh: Whether to apply thresholds to components of the SPEED score.

    Returns:
        A tuple containing:
        - t_error (float): Absolute translation error.
        - t_error_norm (float): Normalized translation error.
        - q_error_deg (float): Orientation error in degrees.
        - speed (float): The final combined SPEED score.
        - accuracy (float): 1.0 if pose is within thresholds, 0.0 otherwise.
    """
    t_error, t_error_norm = calculate_translation_error(t_pred, t_gt)
    q_error_deg, q_error_rad = calculate_orientation_error(q_pred, q_gt)

    speed_t_component = t_error_norm
    speed_q_component = q_error_rad

    if apply_thresh:
        if speed_q_component < ROTATION_THRESHOLD_RAD:
            speed_q_component = 0.0
        if speed_t_component < TRANSLATION_THRESHOLD_NORM:
            speed_t_component = 0.0

    speed = speed_t_component + speed_q_component

    is_accurate = (q_error_rad < ROTATION_THRESHOLD_RAD) and (
            t_error_norm < TRANSLATION_THRESHOLD_NORM
    )
    accuracy = 1.0 if is_accurate else 0.0

    return t_error, t_error_norm, q_error_deg, speed, accuracy


class PoseEstimator:
    """Estimates object pose from 2D-3D point correspondences."""

    def __init__(self, model_coordinates: List, camera_matrix: List):
        """
        Initializes the PoseEstimator.

        Args:
            model_coordinates: List of 3D model points [x, y, z].
            camera_matrix: 3x3 camera intrinsic matrix.
        """
        self.model_points = np.array(model_coordinates, dtype=np.float32)
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def solve_pose(
            self,
            image_points: np.ndarray,
            point_indices: List[int],
            use_ransac: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves for pose using OpenCV's solvePnP.

        Args:
            image_points: Array of detected 2D keypoints in the image.
            point_indices: Indices of the keypoints to use for pose estimation.
            use_ransac: Whether to use the RANSAC scheme.

        Returns:
            A tuple containing:
            - q_xyzw (np.ndarray): The estimated orientation as a (x, y, z, w) quaternion.
            - t_vec (np.ndarray): The estimated translation vector.
        """
        if len(point_indices) < MIN_PNP_POINTS:
            raise ValueError(f"At least {MIN_PNP_POINTS} points are required for solvePnP.")

        model_pts = self.model_points[point_indices]
        image_pts = image_points[point_indices]

        pnp_method = cv2.SOLVEPNP_ITERATIVE

        if use_ransac:
            _, rvec, tvec, _ = cv2.solvePnPRansac(
                model_pts, image_pts, self.camera_matrix, self.dist_coeffs, flags=pnp_method
            )
            # Refine the pose using Levenberg-Marquardt
            rvec, tvec = cv2.solvePnPRefineLM(
                model_pts, image_pts, self.camera_matrix, self.dist_coeffs, rvec, tvec
            )
        else:
            _, rvec, tvec = cv2.solvePnP(
                model_pts, image_pts, self.camera_matrix, self.dist_coeffs, flags=pnp_method
            )

        # Convert rotation vector to quaternion (x, y, z, w)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        q_xyzw = R.from_matrix(rotation_matrix).as_quat()

        return q_xyzw, tvec.flatten()


def run_pose_evaluation(args: argparse.Namespace):
    """
    Main function to run the pose evaluation process.
    """
    # --- Load Data ---
    with open(args.camera, 'r') as f:
        cam_config = json.load(f)
    with open(args.points, 'r') as f:
        model_points_config = json.load(f)

    pred_df = pd.read_json(args.pred)
    gt_df = pd.read_json(args.gt)

    # --- Initialize ---
    pose_estimator = PoseEstimator(model_points_config["keypoints"], cam_config["cameraMatrix"])
    pose_metrics = []

    # --- Merge GT and Predictions for efficient processing ---
    # This is much faster than looping and searching for each file.
    merged_df = pd.merge(gt_df, pred_df, on="filename", suffixes=("_gt", "_pred"))

    print(f"Found {len(merged_df)} matching predictions for {len(gt_df)} ground truth entries.")

    # --- Process Each Image ---
    for row in tqdm(merged_df.itertuples(), total=len(merged_df), desc="Evaluating Poses"):
        t_gt = row.r_xyz
        q_gt = row.q_xyzw

        try:
            # --- Keypoint Selection ---
            # Assumes keypoints are stored as a list of [x, y, confidence]
            keypoints_pred = np.array(row.keypoints_pred, dtype=np.float32)
            keypoints_pred = keypoints_pred.T.reshape([-1, 2]).astype('float32')
            # if keypoints_pred.ndim != 2 or keypoints_pred.shape[1] != 3:
            #     warnings.warn(f"Skipping {row.filename} due to unexpected keypoint shape: {keypoints_pred.shape}")
            #     continue

            image_coords = keypoints_pred[:, :2]
            confidences = np.array(row.confidences, dtype=np.float32)
            # confidences = np.array([random.uniform(0.7, 1.0) for _ in range(10)], dtype='float32')

            # Filter by confidence and select top N points
            confident_indices = np.where(confidences > 0.01)[0]
            top_indices = sorted(confident_indices, key=lambda i: confidences[i], reverse=True)[:args.top_k_points]

            # --- Pose Estimation ---
            if len(top_indices) >= MIN_PNP_POINTS:
                q_pred, t_pred = pose_estimator.solve_pose(
                    image_coords, top_indices, use_ransac=True
                )
            else:
                warnings.warn(
                    f"Not enough confident keypoints for {row.filename} ({len(top_indices)} found). "
                    f"Assigning default error pose.")
                q_pred = np.array([0.0, 0.0, 0.0, 1.0])
                t_pred = np.array([0.0, 0.0, 1.0])

        except Exception as e:
            warnings.warn(f"Pose estimation failed for {row.filename}: {e}. Assigning default error pose.")
            q_pred = np.array([0.0, 0.0, 0.0, 1.0])
            t_pred = np.array([0.0, 0.0, 1.0])

        # --- Metric Calculation ---
        loc_err, norm_loc_err, ori_err_deg, speed, acc = calculate_speed_score(
            t_pred, q_pred, t_gt, q_gt
        )
        pose_metrics.append({
            "filename": row.filename,
            "loc_err": loc_err,
            "norm_loc_err": norm_loc_err,
            "ori_err": ori_err_deg,
            "speed_score": speed,
            "accuracy": acc,
        })

    # --- Save and Display Results ---
    if not pose_metrics:
        print("No poses were evaluated. Exiting.")
        return

    results_df = pd.DataFrame(pose_metrics)
    results_file = args.pred.replace(".json", "_pose_metrics.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    print("\n--- Average Metrics ---")
    print(f"Location Error (m):      {results_df['loc_err'].mean():.4f}")
    print(f"Normalized Loc Error:    {results_df['norm_loc_err'].mean():.4f}")
    print(f"Orientation Error (deg): {results_df['ori_err'].mean():.4f}")
    print(f"SPEED Score:             {results_df['speed_score'].mean():.4f}")
    print(f"Accuracy (within threshold): {results_df['accuracy'].mean() * 100:.2f}%")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate 6D object pose estimation results.")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth labels JSON file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to the prediction labels JSON file.")
    parser.add_argument("--camera", type=str, required=True, help="Path to the camera intrinsics JSON file.")
    parser.add_argument("--points", type=str, required=True, help="Path to the 3D model keypoints JSON file.")
    parser.add_argument("--top_k_points", type=int, default=5,
                        help="Number of top confident keypoints to use for PnP-RANSAC.")
    return parser.parse_args()


if __name__ == "__main__":
    # Suppress specific warnings if needed, but it's often better to see them.
    # warnings.filterwarnings("ignore")
    args = parse_arguments()
    run_pose_evaluation(args)
