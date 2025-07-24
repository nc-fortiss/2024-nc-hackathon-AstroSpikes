from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf


# ==============================================================================
# TensorFlow Implementation of DSNT functions
# ==============================================================================

def _validate_batched_image_tensor_input(tensor: tf.Tensor) -> None:
    """Helper to validate input tensor shape."""
    if not isinstance(tensor, tf.Tensor):
        raise TypeError(f"Input type is not a tf.Tensor. Got {type(tensor)}")
    # Kornia assumes NCHW: (B, C, H, W). We will check for rank 4.
    tf.debugging.assert_rank(
        tensor, 4, message=f"Input tensor must be 4-dimensional (B, C, H, W), but got rank {tf.rank(tensor)}"
    )


def spatial_softmax2d(input_tensor: tf.Tensor, temperature: Optional[tf.Tensor] = None) -> tf.Tensor:
    r"""Apply the Softmax function over features in each image channel.

    Note: The input tensor is assumed to be in NCHW format: (B, C, H, W).

    Args:
        input_tensor: the input tensor with shape :math:`(B, N, H, W)`.
        temperature: factor to apply to input, adjusting the "smoothness" of the output distribution.

    Returns:
       a 2D probability distribution per image channel with shape :math:`(B, N, H, W)`.
    """
    _validate_batched_image_tensor_input(input_tensor)

    shape = tf.shape(input_tensor)
    batch_size, channels, height, width = shape[0], shape[1], shape[2], shape[3]

    if temperature is None:
        temperature = tf.constant(1.0, dtype=input_tensor.dtype)
    else:
        temperature = tf.cast(temperature, dtype=input_tensor.dtype)

    x = tf.reshape(input_tensor, (batch_size, channels, -1))
    x_soft = tf.nn.softmax(x * temperature, axis=-1)

    return tf.reshape(x_soft, (batch_size, channels, height, width))


def spatial_expectation2d(input_tensor: tf.Tensor, normalized_coordinates: bool = True) -> tf.Tensor:
    r"""Compute the expectation of coordinate values using spatial probabilities.

    The input heatmap is assumed to represent a valid spatial probability distribution.
    Note: The input tensor is assumed to be in NCHW format: (B, C, H, W).

    Args:
        input_tensor: the input tensor representing dense spatial probabilities with shape :math:`(B, N, H, W)`.
        normalized_coordinates: whether to return the coordinates normalized in the range
          of :math:`[-1, 1]`. Otherwise, it will return the coordinates in the range of the input shape.

    Returns:
       expected value of the 2D coordinates with shape :math:`(B, N, 2)`. Output order is (x, y).
    """
    _validate_batched_image_tensor_input(input_tensor)

    shape = tf.shape(input_tensor)
    batch_size, channels, height, width = shape[0], shape[1], shape[2], shape[3]

    # Create coordinates grid.
    grid = create_meshgrid_tf(height, width, normalized_coordinates, dtype=input_tensor.dtype)

    # grid is (1, H, W, 2). pos_x/pos_y are (H*W,).
    pos_x = tf.reshape(grid[..., 0], [-1])
    pos_y = tf.reshape(grid[..., 1], [-1])

    # input_tensor is (B, C, H, W). Flatten to (B, C, H*W).
    input_flat = tf.reshape(input_tensor, (batch_size, channels, -1))

    # Compute the expectation of the coordinates.
    # The shapes are: pos_y(H*W) * input_flat(B, C, H*W) -> (B, C, H*W)
    expected_y = tf.reduce_sum(pos_y * input_flat, axis=-1, keepdims=True)
    expected_x = tf.reduce_sum(pos_x * input_flat, axis=-1, keepdims=True)

    output = tf.concat([expected_x, expected_y], axis=-1)

    return tf.reshape(output, (batch_size, channels, 2))  # BxNx2


def _safe_zero_division(numerator: tf.Tensor, denominator: tf.Tensor, eps: float = 1e-32) -> tf.Tensor:
    return numerator / tf.maximum(denominator, eps)


def render_gaussian2d(mean: tf.Tensor, std: tf.Tensor, size: Tuple[int, int],
                      normalized_coordinates: bool = True) -> tf.Tensor:
    r"""Render the PDF of a 2D Gaussian distribution.

    NOTE: This implementation corrects a bug in the original PyTorch code, where
    the standard deviation (`std`) was used in the denominator instead of the
    variance (`std**2`).

    Args:
        mean: the mean location of the Gaussian to render, :math:`(\mu_x, \mu_y)`. Shape: :math:`(*, 2)`.
        std: the standard deviation of the Gaussian to render, :math:`(\sigma_x, \sigma_y)`.
          Shape :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        size: the (height, width) of the output image.
        normalized_coordinates: whether ``mean`` and ``std`` are assumed to use coordinates normalized
          in the range of :math:`[-1, 1]`.

    Returns:
        tensor including rendered points with shape :math:`(*, H, W)`.
    """
    height, width = size

    # Create coordinates grid.
    grid = create_meshgrid_tf(height, width, normalized_coordinates, dtype=mean.dtype)
    pos_x = grid[0, :, :, 0]  # Shape: (H, W)
    pos_y = grid[0, :, :, 1]  # Shape: (H, W)

    # Gaussian PDF = exp(-(x - \mu)^2 / (2 \sigma^2))
    # Add new axes for broadcasting mean/std over the HxW grid.
    mean_x = mean[..., 0, tf.newaxis, tf.newaxis]
    mean_y = mean[..., 1, tf.newaxis, tf.newaxis]
    std_x_sq = tf.square(std[..., 0, tf.newaxis, tf.newaxis])
    std_y_sq = tf.square(std[..., 1, tf.newaxis, tf.newaxis])

    # dists <- (x - \mu)^2
    dist_x = tf.square(pos_x - mean_x)
    dist_y = tf.square(pos_y - mean_y)

    # Assemble the 2D Gaussian.
    # using the corrected formula with variance (std**2)
    exps_x = tf.exp(-dist_x / (2 * std_x_sq))
    exps_y = tf.exp(-dist_y / (2 * std_y_sq))
    gauss = exps_x * exps_y

    # Rescale so that values sum to one.
    val_sum = tf.reduce_sum(gauss, axis=[-2, -1], keepdims=True)
    gauss = _safe_zero_division(gauss, val_sum)

    return gauss


def create_meshgrid_tf(
        height: int, width: int, normalized_coordinates: bool = True, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Generate a coordinate grid for an image.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize coordinates in the range :math:`[-1,1]`.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)` with (x, y) coordinates.
    """
    xs = tf.linspace(0.0, tf.cast(width - 1, tf.float32), width)
    ys = tf.linspace(0.0, tf.cast(height - 1, tf.float32), height)

    if normalized_coordinates:
        xs = (xs / (tf.cast(width - 1, tf.float32))) * 2.0 - 1.0
        ys = (ys / (tf.cast(height - 1, tf.float32))) * 2.0 - 1.0

    x_grid, y_grid = tf.meshgrid(xs, ys, indexing='xy')

    grid = tf.stack([x_grid, y_grid], axis=-1)  # Shape: (H, W, 2)
    grid = tf.expand_dims(grid, axis=0)  # Shape: (1, H, W, 2)
    return tf.cast(grid, dtype=dtype)


def normalize_pixel_coordinates(coords: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """
    Normalize keypoint coordinates from pixel space to the range [-1, 1].

    Args:
        coords: A tensor of pixel coordinates.
                Expected shape: (..., 2) with the last dimension being (x, y).
        height: The height of the image frame.
        width: The width of the image frame.

    Returns:
        A tensor of the same shape as `coords` with coordinates normalized to [-1, 1].
    """
    dtype = coords.dtype

    # CORRECTED LOGIC: Create tensors from Python numbers first.
    scale_dims = tf.constant([width - 1.0, height - 1.0], dtype=dtype)

    scale = tf.constant([2.0, 2.0], dtype=dtype) / scale_dims
    shift = tf.constant([-1.0, -1.0], dtype=dtype)

    # Broadcasting takes care of applying the transformation along the last axis.
    return coords * scale + shift


def denormalize_pixel_coordinates(coords: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """
    Denormalize keypoint coordinates from the range [-1, 1] to pixel space.
    Args:
        coords: A tensor of normalized coordinates.
                Expected shape: (..., 2) with the last dimension being (x, y).
        height: The height of the image frame.
        width: The width of the image frame.

    Returns:
        A tensor of the same shape as `coords` with coordinates in pixel space,
        ranging from [0, width-1] for x and [0, height-1] for y.
    """
    dtype = coords.dtype

    # The inverse transformation is: pixel = ((normalized + 1) / 2) * (dim - 1)
    # This can be done efficiently with broadcasting.
    scale = tf.constant([0.5, 0.5], dtype=dtype)

    # CORRECTED LINE: Create the dims tensor from Python numbers directly.
    dims = tf.constant([width - 1.0, height - 1.0], dtype=dtype)

    return (coords + 1.0) * scale * dims


# ==============================================================================
# Loss Functions
# ==============================================================================

def heatmap_kl_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the KL-Divergence loss between predicted heatmaps and ground truth coordinates.

    This loss function is designed for keypoint detection tasks. It performs the following steps:
    1.  Generates a target heatmap (a 2D Gaussian) from the ground truth (x, y) coordinates.
    2.  Normalizes the predicted heatmap logits into a probability distribution using spatial softmax.
    3.  Calculates the Kullback-Leibler divergence between the target and predicted distributions.

    Args:
        y_true: Ground truth keypoint coordinates.
                Expected Shape: (batch_size, num_keypoints, 2) in (x, y) pixel coordinates.
        y_pred: Predicted heatmap logits from the model.
                Expected Shape: (batch_size, height, width, num_keypoints) (NHWC format).

    Returns:
        A scalar tensor representing the mean KL-Divergence loss across all heatmaps in the batch.
    """
    # --- 1. Get shapes and handle data format (NHWC -> NCHW) ---
    pred_shape = tf.shape(y_pred)
    batch_size, height, width, num_keypoints = pred_shape[0], pred_shape[1], pred_shape[2], pred_shape[3]

    # Transpose predictions from (B, H, W, C) to (B, C, H, W) to match our helpers
    y_pred_nchw = tf.transpose(y_pred, perm=[0, 3, 1, 2])

    # --- 2. Normalize predicted heatmaps to be probability distributions ---
    # The output p_pred is a valid probability distribution per channel.
    p_pred = spatial_softmax2d(y_pred_nchw)

    # --- 3. Generate target probability distribution from ground truth coordinates ---
    # The standard deviation of the target Gaussian is a key hyperparameter.
    # It controls how "sharp" the target is. A smaller stddev is a harder target.
    std_pixel = tf.constant([5.0, 5.0], dtype=y_true.dtype)  # (std_x, std_y) in pixels

    # `render_gaussian2d` expects mean shape (*, 2). y_true is (B, C, 2) which is perfect.
    # It will generate a heatmap of shape (B, C, H, W).
    p_true = render_gaussian2d(
        mean=y_true,
        std=std_pixel,
        size=(height, width),
        normalized_coordinates=False  # Assuming y_true is in pixel coordinates
    )

    # --- 4. Calculate KL Divergence ---
    # KLD expects inputs to be (y_true, y_pred).
    # We need to flatten the spatial dimensions for the loss function.
    # Reshape from (B, C, H, W) -> (B * C, H * W)
    p_true_flat = tf.reshape(p_true, (batch_size * num_keypoints, -1))
    p_pred_flat = tf.reshape(p_pred, (batch_size * num_keypoints, -1))

    # Use the standard Keras KLD loss. It computes sum(p_true * log(p_true / p_pred)).
    kld_loss_fn = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
    loss = kld_loss_fn(p_true_flat, p_pred_flat)

    # Return the mean loss for the batch.
    return loss / tf.cast(batch_size, loss.dtype)


# ==============================================================================
# Metrics
# ==============================================================================

def mpjpe_from_heatmap(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Mean Per Keypoint Position Error (MPJPE) metric for heatmap predictions.

    This function is designed to be used as a Keras metric. It performs the following:
    1.  Converts the predicted heatmap logits into coordinates using the DSNT pipeline
        (spatial softmax followed by spatial expectation).
    2.  Calculates the Euclidean distance between the predicted coordinates and the ground truth coordinates.
    3.  Averages this distance over all keypoints and all items in the batch.

    Args:
        y_true: Ground truth keypoint coordinates.
                Expected Shape: (batch_size, num_keypoints, 2) in (x, y) pixel coordinates.
        y_pred: Predicted heatmap logits from the model.
                Expected Shape: (batch_size, height, width, num_keypoints) (NHWC format).

    Returns:
        A scalar tensor representing the mean pixel error.
    """
    # --- 1. Get predicted coordinates from the heatmap ---
    # Transpose predictions from (B, H, W, C) to (B, C, H, W) to match our helpers
    y_pred_nchw = tf.transpose(y_pred, perm=[0, 3, 1, 2])

    # Apply spatial softmax to get a probability distribution
    p_pred = spatial_softmax2d(y_pred_nchw)

    # Calculate the expected value of the coordinates (in pixel space)
    coords_pred = spatial_expectation2d(p_pred, normalized_coordinates=False)

    # --- 2. Calculate the Euclidean distance error ---
    # y_true and coords_pred are both of shape (batch_size, num_keypoints, 2)
    # tf.norm calculates the L2 norm along the last axis, which is the Euclidean distance.
    # The result is a tensor of distances of shape (batch_size, num_keypoints).
    distances = tf.norm(y_true - coords_pred, axis=-1)

    # --- 3. Compute the mean error ---
    # Take the mean over all keypoints and all batch items to get a single scalar value.
    mean_error = tf.reduce_mean(distances)

    return mean_error
