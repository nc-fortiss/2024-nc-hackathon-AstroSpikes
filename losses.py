import tensorflow as tf

# @keras.saving.register_keras_serializable()
class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_pose=1., lambda_quat=1., lambda_norm=1., reduction=tf.keras.losses.Reduction.AUTO, name='pose_estimation_loss'):
        super(PoseEstimationLoss, self).__init__(reduction=reduction, name=name)
        self.lambda_pose = lambda_pose
        self.lambda_quat = lambda_quat
        self.lambda_norm = lambda_norm
        self.mse = tf.keras.losses.MeanSquaredError(reduction=reduction)

    def call(self, y_true, y_pred):
        # Split the predictions and targets into pose and quaternion parameters
        # Extract first 3 values for the pose
        pred_pose = y_pred[:, :3]  # First 3 values for pose
        # Extract last 4 values for the quaternion
        pred_quat = y_pred[:, 3:]  # Last 4 values for quaternion
        
        # Extract first 3 values for the pose in the target
        target_pose = y_true[:, :3]
        # Extract last 4 values for the quaternion in the target
        target_quat = y_true[:, 3:]
        
        # Normalize the predicted quaternion
        pred_quat_norm = pred_quat / tf.norm(pred_quat, axis=1, keepdims=True)
        
        # Pose estimation loss (Mean Squared Error)
        pose_loss = self.mse(target_pose, pred_pose)
        
        # Quaternion regression loss (Mean Squared Error between normalized quaternions)
        quat_loss = self.mse(target_quat, pred_quat_norm)
        
        # Quaternion normalization loss
        quat_norm_loss = tf.reduce_mean((tf.norm(pred_quat, axis=1) - 1) ** 2)
        
        # Total loss with individual weights
        total_loss = (
            self.lambda_pose * pose_loss +
            self.lambda_quat * quat_loss +
            self.lambda_norm * quat_norm_loss
        )
        
        return total_loss

