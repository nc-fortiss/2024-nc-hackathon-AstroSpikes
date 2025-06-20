import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from dsnt import dsnt

def position_mse_loss(target_pos, pred_pos):
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(target_pos, pred_pos)
    return mse_loss


def geodesic_dist(y_true, y_pred):
    y_pred = tfgt.quaternion.normalize(y_pred)
    return tfgt.quaternion.relative_angle(y_true, y_pred)



def mpkpe_heatmap(y_true, y_pred):
    """
    Computes the Mean Relative L2 Error for translation vectors (x, y, z).
    """
    distance = 0
    #target_pose = tf.reshape(y_true, [y_pred.shape[0],2,y_pred.shape[-1]])
    for ch in range(y_pred.shape[-1]):
        y_pred_coords = dsnt.dsnt(y_pred[:,:,:,ch])[1]
        #y_true = tf.reshape(y_true,y_pred.shape)
        # Compute the L2 norm of the error
        distance += tf.norm(y_true[:,:,ch] - y_pred_coords, axis=-1)  # Shape: (batch_size,)

    # Keras will automatically compute the mean over the batch
    print("distance : ", tf.math.reduce_sum(distance))
    return tf.math.reduce_sum(distance)  # Shape: (batch_size,)

def mpkpe_regression(y_true, y_pred):
    """
    Computes the Mean Relative L2 Error for translation vectors (x, y, z).
    """
 
    # Keras will automatically compute the mean over the batch
    return tf.norm(y_true - y_pred, axis=-1) 


def rel_l2_error(y_true, y_pred):
    """
    Computes the Mean Relative L2 Error for translation vectors (x, y, z).
    """
    # Compute the L2 norm of the error
    error_norm = tf.norm(y_true - y_pred, axis=-1)  # Shape: (batch_size,)

    # Compute the L2 norm of the ground truth
    gt_norm = tf.norm(y_true, axis=-1)  # Shape: (batch_size,)

    # Avoid division by zero
    relative_error = tf.math.divide_no_nan(error_norm, gt_norm)

    # Keras will automatically compute the mean over the batch
    return relative_error  # Shape: (batch_size,)


def ori_error(y_true, y_pred):
    """
    Calculates the orientation error (angular difference) in radians.
    """
    y_pred = tf.linalg.normalize(y_pred, axis=-1)[0]

    # Calculate the dot product between pairs of quaternions in the batch
    dot_product = tf.reduce_sum(y_pred * y_true, axis=-1)

    # Take the absolute value (handles q and -q ambiguity)
    abs_dot_product = tf.abs(dot_product)

    # Clip the value to [0, 1] for numerical stability before acos
    clipped_dot_product = tf.clip_by_value(abs_dot_product, 0.0, 1.0)

    # Calculate the angle (half the total rotation)
    angle_rad = tf.acos(clipped_dot_product)

    # Double the angle to get the full orientation error in radians
    error_rad = 2.0 * angle_rad

    # Return per-sample error. Keras handles the reduction (mean).
    return error_rad
