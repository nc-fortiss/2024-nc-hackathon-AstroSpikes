import tensorflow as tf

# from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
import tensorflow_graphics.geometry.transformation as tfgt


def position_l2_loss(target_pos, pred_pos):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred_pos, target_pos))))


def position_mse_loss(target_pos, pred_pos):
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(target_pos, pred_pos)
    return mse_loss


def quat_rel_angle(y_true, y_pred):
    """Calculates the geodesic distance between two rotation matrices."""
    y_pred = tf.linalg.normalize(y_pred, axis=-1)[0]
    return tfgt.quaternion.relative_angle(y_true, y_pred)


def geodesic_dist(y_true, y_pred):
    y_pred = tfgt.quaternion.normalize(y_pred)
    return tfgt.quaternion.relative_angle(y_true, y_pred)


# def geodesic_loss(target_quat, pred_quat):
#     """Calculates the geodesic distance between two rotation matrices."""
#     # pred_quat = y_pred[:, 3:]
#     # target_quat = y_true[:, 3:]
#     pred_quat = tf.linalg.normalize(pred_quat, axis=-1)[0]
#     eps = tf.constant(1e-7, dtype=tf.float32)
#     m1 = tfgt.rotation_matrix_3d.from_quaternion(target_quat)
#     m2 = tfgt.rotation_matrix_3d.from_quaternion(pred_quat)
#     # Perform batch matrix multiplication
#     m = tf.matmul(m1, tf.transpose(m2, perm=[0, 2, 1]))
#
#     # Calculate cos(theta) using the trace
#     cos = (tf.linalg.trace(m) - 1) / 2.0
#
#     # Clamp cos(theta) to avoid NaN values from tf.acos
#     cos = tf.clip_by_value(cos, -1.0 + eps, 1.0 - eps)
#
#     # Calculate theta (the angle of rotation)
#     theta = tf.acos(cos)
#
#     # Return the mean angle (Geodesic Loss)
#     return tf.reduce_mean(theta)


class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, name="pose_estimation_loss"):
        super(PoseEstimationLoss, self).__init__(name=name)
        self.alpha = tf.Variable(0.1, trainable=True, dtype=tf.float32,
                                 constraint=lambda x: tf.clip_by_value(x, 0.01, 1.0))

    @tf.function
    def call(self, y_true, y_pred):
        """
        Calculates the combined position and quaternion loss, optimized for efficiency.
        """
        # Unpack position and quaternion from outputs

        # **Position Loss (Mean Squared Error)**
        pred_pos = y_pred[:, :3]
        target_pos = y_true[:, :3]
        pos_loss = position_l2_loss(target_pos, pred_pos)

        # **Quaternion Loss**
        pred_quat = y_pred[:, 3:]
        target_quat = y_true[:, 3:]
        rot_loss = geodesic_loss(target_quat, pred_quat)

        total_loss = self.alpha * pos_loss + (1 - self.alpha) * rot_loss
        # total_loss = pos_loss + rot_loss

        return total_loss
