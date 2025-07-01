"""
A Tensorflow implementation of the DSNT layer, as taken from the paper "Numerical Coordinate
Regression with Convolutional Neural Networks". This version includes corrections and
modernizations for clarity and performance.
"""

import tensorflow as tf


def dsnt(inputs, method='softmax'):
    """
    Differentiable Spatial to Numerical Transform, as taken from the paper "Numerical Coordinate
    Regression with Convolutional Neural Networks"
    Arguments:
        inputs - The learnt heatmap. A 4d tensor of shape [batch, height, width, 1].
        method - A string representing the normalisation method. See `_normalise_heatmap` for available methods.
    Returns:
        norm_heatmap - The given heatmap with normalisation/rectification applied. Shape: [batch, height, width].
        coords_zipped - A tensor of shape [batch, 2] containing the [x, y] coordinate pairs.
    """
    # Rectify and reshape inputs
    norm_heatmap = _normalise_heatmap(inputs, method)

    # Get shapes
    batch_count = tf.shape(norm_heatmap)[0]
    height = tf.shape(norm_heatmap)[1]
    width = tf.shape(norm_heatmap)[2]

    # Build the DSNT x, y matrices
    # Create normalized coordinate vectors
    range_x = tf.cast(tf.range(width), tf.float32)
    range_y = tf.cast(tf.range(height), tf.float32)

    # The paper's formula is (2k - (N+1))/N for k in [1, N]
    # For 0-indexed k in [0, N-1], this is (2k - N + 1)/N
    dsnt_x_coords = (2.0 * range_x - tf.cast(width, tf.float32) + 1.0) / tf.cast(width, tf.float32)
    dsnt_y_coords = (2.0 * range_y - tf.cast(height, tf.float32) + 1.0) / tf.cast(height, tf.float32)

    # Tile to create coordinate grids
    dsnt_x = tf.tile(tf.reshape(dsnt_x_coords, [1, 1, width]), [batch_count, height, 1])
    dsnt_y = tf.tile(tf.reshape(dsnt_y_coords, [1, height, 1]), [batch_count, 1, width])

    # Compute the Frobenius inner product
    outputs_x = tf.reduce_sum(norm_heatmap * dsnt_x, axis=[1, 2])
    outputs_y = tf.reduce_sum(norm_heatmap * dsnt_y, axis=[1, 2])

    # Zip into [x, y] pairs
    coords_zipped = tf.stack([outputs_x, outputs_y], axis=1)

    return norm_heatmap, coords_zipped


def _normalise_heatmap(inputs, method='softmax'):
    """
    Applies the chosen normalisation/rectification method to the input tensor.
    Arguments:
        inputs - A tensor of shape [batch, height, width] or [batch, height, width, 1].
        method - A string representing the normalisation method. One of those shown below.
    """
    # If the input is 4D, remove the final dimension
    if len(inputs.shape) == 4:
        inputs = tf.reshape(inputs, tf.shape(inputs)[:3])

    # Normalise the values such that the values sum to one for each heatmap
    # Using the / operator is more modern than tf.div
    normalise = lambda x: x / tf.reshape(tf.reduce_sum(x, [1, 2]), [-1, 1, 1])

    # Perform rectification
    if method == 'softmax':
        inputs = _softmax2d(inputs, axes=[1, 2])
    elif method == 'abs':
        inputs = tf.abs(inputs)
        inputs = normalise(inputs)
    elif method == 'relu':
        inputs = tf.nn.relu(inputs)
        inputs = normalise(inputs)
    elif method == 'sigmoid':
        inputs = tf.nn.sigmoid(inputs)
        inputs = normalise(inputs)
    else:
        msg = f"Unknown rectification method \"{method}\""
        raise ValueError(msg)
    return inputs


def _kl_2d(p, q, eps=1e-24):
    unsummed_kl = p * (tf.math.log(p + eps) - tf.math.log(q + eps))
    kl_values = tf.reduce_sum(unsummed_kl, axis=[-1, -2])
    return kl_values


def _js_2d(p, q, eps=1e-24):
    m = 0.5 * (p + q)
    return 0.5 * _kl_2d(p, m, eps) + 0.5 * _kl_2d(q, m, eps)


def _softmax2d(target, axes):
    """
    A softmax implementation which can operate across more than one axis.
    """
    max_axis = tf.reduce_max(target, axes, keepdims=True)
    target_exp = tf.exp(target - max_axis)
    normalize = tf.reduce_sum(target_exp, axes, keepdims=True)
    softmax = target_exp / normalize
    return softmax


def _make_gaussian(size, centre, fwhm=1.0):
    """
    Makes a rectangular gaussian kernel. (More efficient version)
    """
    height, width = size[0], size[1]
    # Scale the normalised coordinates to be relative to the size of the frame
    centre_pix = [centre[0] * tf.cast(width, tf.float32),
                  centre[1] * tf.cast(height, tf.float32)]

    # Create coordinate grids
    x_coords = tf.cast(tf.range(width), tf.float32)
    y_coords = tf.cast(tf.range(height), tf.float32)

    # Use broadcasting to create the 2D Gaussian
    x0 = centre_pix[0] - 0.5
    y0 = centre_pix[1] - 0.5

    # Reshape for broadcasting: y becomes [H, 1], x is [W] (becomes [1, W] automatically)
    y_coords = tf.reshape(y_coords, [height, 1])

    # The gaussian formula
    variance = (fwhm / (2.0 * tf.math.sqrt(2.0 * tf.math.log(2.0)))) ** 2
    unnorm = tf.exp(-((x_coords - x0) ** 2 + (y_coords - y0) ** 2) / (2 * variance))

    # Normalize to sum to 1
    norm = unnorm / tf.reduce_sum(unnorm)
    return norm


def _make_gaussians(centres_in, height, width, fwhm=1.0):
    """
    Makes a batch of gaussians using tf.map_fn for better performance and readability.
    """

    # The function to apply to each centre coordinate
    def make_gaussian_for_map(centre):
        return _make_gaussian([height, width], centre, fwhm)

    # Use tf.map_fn to apply the function over the batch of centres
    heatmaps_out = tf.map_fn(make_gaussian_for_map, centres_in, dtype=tf.float32)
    return heatmaps_out


def js_reg_loss(heatmaps, centres, fwhm=1.0):
    """
    Calculates and returns the average Jensen-Shannon divergence between heatmaps and target Gaussians.
    Arguments:
        heatmaps - Heatmaps generated by the model. Shape: [batch, height, width].
        centres - Centres of the target Gaussians (in normalized units). Shape: [batch, 2].
        fwhm - Full-width-half-maximum for the drawn Gaussians, which can be thought of as a radius.
    """
    # Ensure heatmaps are normalized
    heatmaps_norm = _normalise_heatmap(heatmaps)

    gauss = _make_gaussians(centres, tf.shape(heatmaps_norm)[1], tf.shape(heatmaps_norm)[2], fwhm)
    divergences = _js_2d(heatmaps_norm, gauss)
    return tf.reduce_mean(divergences)


def kl_reg_loss(heatmaps, centres, fwhm=1.0):
    """
    Calculates and returns the average Kulbeck-leiber divergence between heatmaps and target Gaussians.
    Arguments:
        heatmaps - Heatmaps generated by the model. Shape: [batch, height, width].
        centres - Centres of the target Gaussians (in normalized units). Shape: [batch, 2].
        fwhm - Full-width-half-maximum for the drawn Gaussians, which can be thought of as a radius.
    """
    # Ensure heatmaps are normalized
    heatmaps_norm = _normalise_heatmap(heatmaps)

    gauss = _make_gaussians(centres, tf.shape(heatmaps_norm)[1], tf.shape(heatmaps_norm)[2], fwhm)
    divergences = _kl_2d(heatmaps_norm, gauss)
    return tf.reduce_mean(divergences)
