from scipy.stats import multivariate_normal
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from src.utils import dsnt


def build_heatmap(target_pos):
    """
    For a list of target keypoints, generate one gaussian heatmap per keypoint
    """
    target_keypoints = target_pos.reshape(3, -1)
    keypoints_heatmaps = np.array((8, 56, 56))
    pos = np.dstack(np.mgrid[0:56:1, 0:56:1])

    for i, keypoint in enumerate(target_keypoints):
        rv = multivariate_normal(mean=[keypoint[0], keypoint[1]], cov=4)
        if keypoint[2]:  # if the keypoint is visible
            keypoints_heatmaps[i] = rv.pdf(pos)

    return tf.convert_to_tensor(keypoints_heatmaps)


# def combined_loss(targets, pred):
#     # target_pose = tf.reshape(targets, [pred.shape[0],2,pred.shape[-1]])
#     sum_loss = 0
#     for ch in range(pred.shape[-1]):
#         norm_heatmaps, coords = dsnt.dsnt(pred[:, :, :, ch])
#         sum_loss += tf.losses.mean_squared_error(coords, targets[:, :, ch]) + dsnt.js_reg_loss(norm_heatmaps,
#                                                                                                targets[:, :, ch],
#                                                                                                fwhm=3)
#     print("sum_loss :", tf.math.reduce_sum(sum_loss))
#     return tf.math.reduce_sum(sum_loss)

# def heatmap_loss(y_true, y_pred_logits):
#     # y_true is y_true_coords from your dataset
#     y_true_coords = y_true
#     print(y_true_coords.shape)
#
#     # # --- FIX ---
#     # # Reshape the ground truth coordinates to match the expected input shape for mu_t.
#     # # If y_true_coords has shape (Batch, 2), this will reshape it to (Batch, 1, 2).
#     # # This ensures it has a "Locations" dimension that matches the heatmaps.
#     # if tf.rank(y_true_coords) == 2:
#     #     y_true_coords = tf.expand_dims(y_true_coords, axis=1)
#     # # --- END FIX ---
#
#     # Now the shapes are compatible and the assertion will pass.
#     return dsnt.kl_reg_losses(y_pred_logits, y_true_coords, sigma_t=1.0)

# def heatmap_loss(y_true_coords, y_pred_logits):
#     # heatmaps, mu_t, sigma_t
#     return dsnt.kl_reg_losses(y_pred_logits, y_true_coords, sigma_t=1.0)


# It's good practice to instantiate the loss class once
mse_loss_fn = tf.keras.losses.MeanSquaredError()


def heatmap_loss(y_true, y_pred_logits):
    """
    Calculates heatmap loss using Mean Squared Error (MSE) for stability.
    This is a diagnostic replacement for the unstable KL Divergence loss.
    """
    y_true_coords = y_true

    # --- Reshaping logic (remains the same and is crucial) ---
    if tf.rank(y_pred_logits) == 4:
        pred_shape = tf.shape(y_pred_logits)
        if pred_shape[1] > pred_shape[3]:
            y_pred_logits = tf.transpose(y_pred_logits, perm=[0, 3, 1, 2])
    elif tf.rank(y_pred_logits) == 3:
        y_pred_logits = tf.expand_dims(y_pred_logits, axis=1)

    if tf.rank(y_true_coords) == 2:
        y_true_coords = tf.expand_dims(y_true_coords, axis=1)

    y_true_coords = tf.ensure_shape(y_true_coords, [None, None, 2])

    # --- Loss Calculation Changed to MSE ---

    # 1. Generate the target Gaussian heatmap `q` from the ground truth coordinates.
    #    The `make_gauss` function we built is perfect for this.
    target_heatmap = dsnt.make_gauss(y_true_coords, tf.shape(y_pred_logits)[2:], sigma=1.0)
    target_heatmap = tf.debugging.check_numerics(target_heatmap, "Target heatmap has NaNs")

    # 2. Convert the model's logits into a probability heatmap `p`.
    predicted_heatmap = dsnt.flat_softmax(y_pred_logits)
    predicted_heatmap = tf.debugging.check_numerics(predicted_heatmap, "Predicted heatmap has NaNs")

    # 3. Calculate the Mean Squared Error between the two heatmaps.
    #    This is a much more numerically stable operation than KL divergence.
    # loss = mse_loss_fn(target_heatmap, predicted_heatmap)
    loss = tf.keras.losses.kl_divergence(target_heatmap, predicted_heatmap)

    return loss


def mse_loss(target_pos, pred_pos):
    keypoints_heatmaps = build_heatmap(target_pos)
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(keypoints_heatmaps, pred_pos)
    return mse_loss
