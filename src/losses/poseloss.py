import tensorflow as tf


class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, name="pose_estimation_loss"):
        super(PoseEstimationLoss, self).__init__(name=name)
        self.pos_loss = tf.keras.losses.MeanSquaredError()
        self.alpha = tf.Variable(0.1, trainable=True, dtype=tf.float32,
                                 constraint=lambda x: tf.clip_by_value(x, 0.01, 1.0))

    @tf.function
    def call(self, y_true, y_pred):
        """
        Calculates the combined position and quaternion loss, optimized for efficiency.
        """
        # Unpack position and quaternion from outputs
        pred_pos = y_pred[:, :3]
        pred_quat = y_pred[:, 3:]
        target_pos = y_true[:, :3]
        target_quat = y_true[:, 3:]

        # **Position Loss (Mean Squared Error)**
        pos_loss = self.pos_loss(target_pos, pred_pos)
        # **Quaternion Loss**
        loss_quat = self.quaternion_distance(target_quat, pred_quat)

        return self.alpha * pos_loss + (1 - self.alpha) * loss_quat

    def quaternion_distance(self, q1, q2):
        q2 = tf.linalg.normalize(q2, axis=-1)[0]
        dot_product = tf.abs(tf.reduce_sum(q1 * q2, axis=-1))  # Compute dot product
        distance = 1 - dot_product
        return tf.reduce_mean(distance)
