import keras

from src.utils import dsnt
from src.utils.kdsnt import *


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


@keras.saving.register_keras_serializable()
def dsnt_loss(y_true_kpts, y_pred_htmps_):
    """
    Calculates heatmap loss for a model with a CHANNELS-LAST output (B, H, W, N).
    """
    # Get shape from the channels-last prediction

    y_pred_htmps = tf.transpose(y_pred_htmps_, perm=[0, 3, 1, 2])

    input_shape = tf.shape(y_pred_htmps)
    h = input_shape[2]
    w = input_shape[3]

    num_batches = tf.shape(y_true_kpts)[0]
    num_kpts = tf.shape(y_true_kpts)[1]

    std = tf.constant([0.025, 0.025], dtype=y_true_kpts.dtype)
    std_base = std[tf.newaxis, tf.newaxis, :]
    stdx = tf.tile(std_base, [num_batches, num_kpts, 1])

    # 1. Generate the target Gaussian heatmap.
    target_heatmap = render_gaussian_2d(
        mean=y_true_kpts, std=stdx, size=(h, w), normalized_coordinates=True
    )

    # 2. Your kdsnt functions now correctly handle the (B, N, H, W) prediction
    predicted_heatmap = spatial_softmax2d(y_pred_htmps)
    predicted_kpts = spatial_expectation2d(predicted_heatmap, normalized_coordinates=True)

    # 3. Calculate losses (this part was already correct)
    per_example_kpts_loss = tf.keras.losses.mean_squared_error(y_true_kpts, predicted_kpts)

    # Reshape both to (B, -1) for KL divergence
    per_example_htmps_loss = tf.keras.losses.kl_divergence(
        tf.reshape(target_heatmap, [num_batches, -1]),
        tf.reshape(predicted_heatmap, [num_batches, -1])
    )

    kpts_loss = tf.nn.compute_average_loss(per_example_kpts_loss)
    htmps_loss = tf.nn.compute_average_loss(per_example_htmps_loss)

    loss = kpts_loss + htmps_loss
    return loss


if __name__ == '__main__':
    dummy_true_kpts = tf.random.normal(shape=(16, 8, 2))
    dummy_pred_htmps = tf.random.normal(shape=(16, 56, 56, 8))

    # Call the function directly
    loss_value = dsnt_loss(dummy_true_kpts, dummy_pred_htmps)
    print(f"Loss value: {loss_value}")
