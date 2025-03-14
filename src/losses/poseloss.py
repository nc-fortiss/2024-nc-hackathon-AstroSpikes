import tensorflow as tf


class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, name="pose_estimation_loss"):
        super(PoseEstimationLoss, self).__init__(name=name)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.alpha = tf.Variable(0.5, trainable=True, dtype=tf.float32,
                                 constraint=lambda x: tf.clip_by_value(x, 0.01, 1.0))  # Soft constraint for stability

    @tf.function
    def call(self, y_true, y_pred):
        # Unpack position and quaternion from outputs
        pred_pos = y_pred[:, :3]
        pred_quat = y_pred[:, 3:]

        target_pos = y_true[:, :3]
        target_quat = y_true[:, 3:]

        # **Position Loss (MSE)**
        pose_loss = self.mse(target_pos, pred_pos)

        # **Quaternion Loss (Quaternion Distance)**
        # Ensures quaternion is unit-normalized and invariant to sign flipping
        dot_product = tf.abs(tf.reduce_sum(target_quat * pred_quat, axis=1, keepdims=True))  # |q_true ⋅ q_pred|
        quat_loss = tf.reduce_mean(1 - dot_product)  # Quaternion Distance Loss

        # **Total Loss**
        total_loss = pose_loss + self.alpha * quat_loss
        return total_loss
