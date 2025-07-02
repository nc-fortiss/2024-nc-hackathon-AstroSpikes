import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from src.utils import dsnt


def position_mse_loss(target_pos, pred_pos):
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(target_pos, pred_pos)
    return mse_loss


def geodesic_dist(y_true, y_pred):
    y_pred = tfgt.quaternion.normalize(y_pred)
    return tfgt.quaternion.relative_angle(y_true, y_pred)


def mpkpe_heatmap(y_true, y_pred):
    """
    Mean Per Keypoint Position Error metric for heatmap predictions.

    This metric calculates the Euclidean distance between the predicted coordinates
    (derived from heatmaps) and the ground truth coordinates. It correctly
    handles conversion of normalized ground truth coordinates to pixel space.

    Args:
        y_true (tf.Tensor): Ground truth coordinates in NORMALIZED [-1, 1] format.
        y_pred (tf.Tensor): Predicted heatmaps from the model.

    Returns:
        tf.Tensor: The mean Euclidean error in pixels.
    """
    y_true_coords = y_true
    y_pred_heatmaps = y_pred

    # 1. Ensure y_pred_heatmaps is in channels_first format (B, C, H, W) for dsnt
    if tf.rank(y_pred_heatmaps) == 4:
        pred_shape = tf.shape(y_pred_heatmaps)
        # Heuristic for (B, H, W, C) -> Transpose to (B, C, H, W)
        if pred_shape[1] > pred_shape[3]:
            y_pred_heatmaps = tf.transpose(y_pred_heatmaps, perm=[0, 3, 1, 2])

    # 2. Convert predicted heatmaps to PIXEL coordinates.
    #    The output `y_pred_coords` will have shape (Batch, Locations, 2).
    y_pred_coords = dsnt.dsnt(y_pred_heatmaps, normalized_coordinates=False, method='softmax')

    # 3. Reshape y_true_coords to match y_pred_coords' rank.
    #    This adds the "Locations" dimension if it's missing.
    if tf.rank(y_true_coords) == 2:
        y_true_coords = tf.expand_dims(y_true_coords, axis=1)  # -> (B, 1, 2)

    # 4. *** NEW: Convert y_true from Normalized to Pixel Coordinates ***
    #    To compare with `y_pred_coords`, we must convert the normalized
    #    `y_true_coords` into the same pixel space. We use the heatmap's size for this.
    heatmap_size = tf.shape(y_pred_heatmaps)[2:]  # Gets the [Height, Width]
    y_true_coords = dsnt.normalized_to_pixel_coordinates(y_true_coords, heatmap_size)

    # 5. Calculate the Euclidean distance (MPJPE) in pixel units.
    #    Both tensors should now have shape (B, L, 2) and be in pixel coordinates.
    euclidean_distance = tf.norm(y_true_coords - y_pred_coords, ord='euclidean', axis=-1)

    # Return the mean distance across all batches and locations
    return tf.reduce_mean(euclidean_distance)


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
