import tensorflow as tf


class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, beta, name='pose_estimation_loss'):
        super(PoseEstimationLoss, self).__init__(name=name)
        self.beta = beta  # Beta is now a fixed scaling factor for each run
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        pred_pose = y_pred[:, :3]  # First 3 values for position
        pred_quat = y_pred[:, 3:]  # Last 4 values for quaternion

        target_pose = y_true[:, :3]
        target_quat = y_true[:, 3:]

        epsilon = 1e-7
        # # Normalize the predicted quaternion
        pred_quat_norm = pred_quat / (tf.norm(pred_quat, axis=1, keepdims=True) + epsilon)

        # Compute position loss (MSE)
        pose_loss_tensor = self.mse(target_pose, pred_pose)

        # Compute orientation loss (Quaternion MSE)
        quat_loss_tensor = self.mse(target_quat, pred_quat_norm)

        # Total loss as a linear combination of the two losses with scaling factor beta
        total_loss = pose_loss_tensor + self.beta * quat_loss_tensor

        return total_loss
