from scipy.stats import multivariate_normal
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from src.utils import dsnt
import keras

@keras.saving.register_keras_serializable()
def heatmap_loss(y_true, y_pred_logits):
    """
    Calculates heatmap loss using Mean Squared Error (MSE), ensuring stability
    and correct output shape for Keras.
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

    # 1. Generate the target Gaussian heatmap from the ground truth coordinates.
    target_heatmap = dsnt.make_gauss(y_true_coords, tf.shape(y_pred_logits)[2:], sigma=1.0)

    # 2. Convert the model's logits into a probability heatmap.
    predicted_heatmap = dsnt.flat_softmax(y_pred_logits)

    # 3. Calculate the Mean Squared Error between the two heatmaps.
    #    This is a much more numerically stable operation than KL divergence.
    # loss = mse_loss_fn(target_heatmap, predicted_heatmap)
    loss = tf.keras.losses.kl_divergence(target_heatmap, predicted_heatmap)

    return loss

    # # 3. Calculate the Mean Squared Error and ensure correct output shape.
    # # Calculate squared error for each pixel: shape (B, L, H, W)
    # squared_error = tf.square(target_heatmap - predicted_heatmap)
    #
    # # Reduce over spatial and location dims to get loss per sample: shape (B,)
    # # Axes to reduce: Locations (1), Height (2), Width (3)
    # loss_per_sample = tf.reduce_mean(squared_error, axis=[1, 2, 3])

    # return loss_per_sample
