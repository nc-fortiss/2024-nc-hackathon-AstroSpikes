import numpy as np
import pandas as pd


def compute_pose_score(gt_csv_path, pred_csv_path):
    # pose score based on https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/
    # Load ground truth and prediction CSVs
    gt_df = pd.read_csv(gt_csv_path)
    pred_df = pd.read_csv(pred_csv_path)

    # Merge on 'filepath'
    merged_df = pd.merge(gt_df, pred_df, on='filepath', suffixes=('_gt', '_est'))

    total_pose_score = 0.0
    total_position_error = 0.0
    total_orientation_error = 0.0
    N = len(merged_df)

    for _, row in merged_df.iterrows():
        # Translation vectors
        r_gt = np.array([row['Tx_gt'], row['Ty_gt'], row['Tz_gt']])
        r_est = np.array([row['Tx_est'], row['Ty_est'], row['Tz_est']])

        # Position score: L2 norm / L2 norm of GT
        position_error = np.linalg.norm(r_gt - r_est) / (np.linalg.norm(r_gt) + 1e-8)

        # Quaternions
        q_gt = np.array([row['Qx_gt'], row['Qy_gt'], row['Qz_gt'], row['Qw_gt']])
        q_est = np.array([row['Qx_est'], row['Qy_est'], row['Qz_est'], row['Qw_est']])

        # Orientation score: 2 * arccos(|⟨q_est, q_gt⟩|)
        mmp = np.inner(q_est, q_gt)

        if mmp < -1.0: mmp = -1.0
        if mmp > 1.0: mmp = 1.0

        orientation_error = 2 * np.arccos(np.abs(mmp))
        # dot_product = np.clip(np.abs(np.dot(q_gt, q_est)), 0.0, 1.0)
        # orientation_error = 2 * np.arccos(dot_product)

        # Total pose score for this sample
        pose_score = position_error + orientation_error
        total_pose_score += pose_score
        total_position_error += position_error
        total_orientation_error += orientation_error

    average_pose_score = total_pose_score / N
    average_position_score = total_position_error / N
    average_orientation_score = total_orientation_error / N

    print(f"\nPosition Error: {average_position_score}")
    print(f"\nRotation Error: {average_orientation_score}")
    print(f"\nPose Score: {average_pose_score}")
    return average_position_score, average_orientation_score, average_pose_score
