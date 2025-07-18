from typing import Optional

import tensorflow as tf


def _validate_batched_image_tensor_input(tensor: tf.Tensor) -> None:
    """Validate that the input is a 4D tensor (B, H, W, N)."""
    if not isinstance(tensor, tf.Tensor):
        raise TypeError("Input is not a TensorFlow Tensor.")
    # Check for 4 dimensions
    if len(tensor.shape) != 4:
        raise ValueError(f"Input tensor must be 4D, but got rank {len(tensor.shape)}.")


def spatial_softmax2d(input: tf.Tensor, temperature: Optional[tf.Tensor] = None) -> tf.Tensor:
    r"""Applies Softmax over spatial features for a channels-last input.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        ...
    """
    input_shape = tf.shape(input)
    batch_size, channels, height, width = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

    if temperature is None:
        temperature = tf.constant(1.0, dtype=input.dtype)
    temperature = tf.cast(temperature, dtype=input.dtype)

    # Reshape (B, N, H, W) -> (B, N, H*W)
    x = tf.reshape(input, (batch_size, channels, -1))
    x_soft = tf.nn.softmax(x * temperature, axis=-1)

    # Reshape back to (B, N, H, W)
    output = tf.reshape(x_soft, (batch_size, channels, height, width))

    return output


def spatial_expectation2d(input: tf.Tensor, normalized_coordinates: bool = True) -> tf.Tensor:
    _validate_batched_image_tensor_input(input)

    # Use tf.shape for dynamic compatibility
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Create coordinates grid
    if normalized_coordinates:
        # Cast height and width to float for linspace if they are not
        h_float = tf.cast(height, dtype=tf.float32)
        w_float = tf.cast(width, dtype=tf.float32)
        x_coords = tf.linspace(-1.0, 1.0, width)
        y_coords = tf.linspace(-1.0, 1.0, height)
    else:
        x_coords = tf.range(width, dtype=input.dtype)
        y_coords = tf.range(height, dtype=input.dtype)

    # Create the grid and flatten
    grid_x, grid_y = tf.meshgrid(x_coords, y_coords)
    pos_x = tf.reshape(grid_x, [-1])
    pos_y = tf.reshape(grid_y, [-1])

    # Flatten the input heatmap
    input_flat = tf.reshape(input, (batch_size, channels, -1))

    # Compute the expectation of the coordinates
    # The grids pos_x and pos_y will be broadcasted to match input_flat
    expected_y = tf.reduce_sum(pos_y * input_flat, axis=-1, keepdims=True)
    expected_x = tf.reduce_sum(pos_x * input_flat, axis=-1, keepdims=True)

    # The result is already (B, N, 2)
    return tf.concat([expected_x, expected_y], axis=-1)


def _safe_zero_division(numerator: tf.Tensor, denominator: tf.Tensor, eps: float = 1e-32) -> tf.Tensor:
    return numerator / tf.clip_by_value(denominator, clip_value_min=eps, clip_value_max=tf.reduce_max(denominator))


def render_gaussian_2d(mean: tf.Tensor, std: tf.Tensor, size: tuple[int, int],
                       normalized_coordinates: bool = True) -> tf.Tensor:
    r"""Render the PDF of a 2D Gaussian distribution.

    Args:
        mean: the mean location of the Gaussian to render, :math:`(\mu_x, \mu_y)`. Shape: :math:`(*, 2)`.
        std: the standard deviation of the Gaussian to render, :math:`(\sigma_x, \sigma_y)`.
          Shape :math:`(*, 2)`. Should be able to be broadcast with `mean`.
        size: the (height, width) of the output image.
        normalized_coordinates: whether ``mean`` and ``std`` are assumed to use coordinates normalized
          in the range of :math:`[-1, 1]`. Otherwise, coordinates are assumed to be in the range of the output shape.

    Returns:
        tensor including rendered points with shape :math:`(*, H, W)`.
    """
    if not (std.dtype == mean.dtype):
        raise TypeError("Expected inputs to have the same dtype.")

    height, width = size

    # Create coordinates grid.
    if normalized_coordinates:
        x_coords = tf.linspace(-1.0, 1.0, width)
        y_coords = tf.linspace(-1.0, 1.0, height)
    else:
        x_coords = tf.range(width, dtype=mean.dtype)
        y_coords = tf.range(height, dtype=mean.dtype)

    grid_x, grid_y = tf.meshgrid(x_coords, y_coords)
    pos_x = tf.reshape(grid_x, (height, width))
    pos_y = tf.reshape(grid_y, (height, width))

    # Gaussian PDF = exp(-(x - \mu)^2 / (2 \sigma^2))
    #              = exp(dists * ks),
    #                where dists = (x - \mu)^2 and ks = -1 / (2 \sigma^2)

    # dists <- (x - \mu)^2
    dist_x = tf.square(pos_x - mean[..., 0, None, None])
    dist_y = tf.square(pos_y - mean[..., 1, None, None])

    # ks <- -1 / (2 \sigma^2)
    k_x = -0.5 * tf.math.reciprocal(tf.square(std[..., 0, None, None]))
    k_y = -0.5 * tf.math.reciprocal(tf.square(std[..., 1, None, None]))

    # Assemble the 2D Gaussian.
    exps_x = tf.exp(dist_x * k_x)
    exps_y = tf.exp(dist_y * k_y)
    gauss = exps_x * exps_y

    # Rescale so that values sum to one.
    val_sum = tf.reduce_sum(gauss, axis=[-2, -1], keepdims=True)
    gauss = _safe_zero_division(gauss, val_sum)

    return gauss


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    dummy_true_kpts = tf.random.normal(shape=(16, 8, 2))
    dummy_pred_htmps = tf.random.normal(shape=(16, 8, 56, 56))

    # Call the function directly
    loss_value = dsnt_loss(dummy_true_kpts, dummy_pred_htmps)

    # Generate the heatmap
    mean = tf.constant([[0.5, 0.5]], dtype=tf.float32)
    std = tf.constant([[0.025, 0.025]], dtype=tf.float32)
    heatmap = render_gaussian_2d(mean, std, size=(56, 56), normalized_coordinates=True)

    # Visualize the heatmap. [1, 2, 3]
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap.numpy()[0], cmap='viridis')
    plt.title('Generated 56x56 Heatmap')
    plt.show()
    print(heatmap.shape)
    print(spatial_expectation2d(heatmap, normalized_coordinates=True).numpy())
