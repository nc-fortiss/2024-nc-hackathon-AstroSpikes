# Copyright 2017 Aiden Nibali
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
#
# This file has been converted from the original PyTorch version to TensorFlow.
# Conversion performed by an AI model and subsequently debugged.

"""
DSNT (soft-argmax) operations for use in TensorFlow computation graphs.
"""

from functools import reduce
from operator import mul

import tensorflow as tf


def linear_expectation(probs, values):
    """
    Calculates the expectation of values along spatial dimensions in a graph-compatible way.

    Args:
        probs (tf.Tensor): A tensor of probability distributions (heatmaps), with
            shape (B, C, D1, D2, ...).
        values (list of tf.Tensor): A list of 1D tensors representing the coordinate
            values for each spatial dimension. The length of the list must match
            the number of spatial dimensions in `probs`.

    Returns:
        tf.Tensor: The calculated expectation, with shape (B, C, N), where N is the
            number of spatial dimensions.
    """
    # Use tf.Assert for a graph-compatible runtime assertion.
    num_spatial_dims = tf.rank(probs) - 2
    tf.Assert(tf.equal(len(values), num_spatial_dims),
              ["The number of value vectors must match the number of spatial dimensions in probs."])

    expectation = []
    # len(values) is a static Python integer, so this loop is safe.
    for i in range(len(values)):
        # The axis of the current spatial dimension we are calculating the expectation for.
        current_spatial_axis = i + 2

        # To calculate the expectation along one axis, we need to marginalize the
        # probability distribution by summing over all *other* spatial axes.
        other_spatial_axes = [j + 2 for j in range(len(values)) if i != j]

        # `marg` has its other spatial dimensions summed out, leaving a shape like
        # (B, C, D_i), where D_i is the size of the current spatial dimension.
        marg = tf.reduce_sum(probs, axis=other_spatial_axes)

        # The coordinate values for the current dimension.
        value_vec = values[i]

        # Calculate the expectation: E[x] = sum(p(x) * x)
        # `value_vec` will broadcast correctly to the shape of `marg`.
        # The axis of interest (e.g., H or W) is now the last axis of `marg`.
        exp = tf.reduce_sum(marg * value_vec, axis=-1)
        expectation.append(exp)

    # Stack the expectations for each dimension to get the final coordinates.
    return tf.stack(expectation, axis=-1)


def normalized_linspace(length, dtype=tf.float32):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector (int or scalar tf.Tensor).
        dtype: The dtype of the resulting tensor.

    Returns:
        The generated vector.
    """
    length = tf.cast(length, dtype)
    first = -(length - 1.0) / length
    return tf.range(length, dtype=dtype) * (2.0 / length) + first


def soft_argmax(heatmaps, normalized_coordinates=True):
    """Computes the soft-argmax of heatmaps.

    Args:
        heatmaps (tf.Tensor): A tensor of heatmaps, with shape (B, C, H, W, ...).
        normalized_coordinates (bool): If True, returns coordinates in the [-1, 1] range.
            Otherwise, returns pixel coordinates.

    Returns:
        A tensor of coordinates, with shape (B, C, N), where N is the number of
        spatial dimensions.
    """
    spatial_dims = tf.shape(heatmaps)[2:]

    if normalized_coordinates:
        values = [normalized_linspace(d, dtype=heatmaps.dtype)
                  for d in tf.unstack(spatial_dims)]
    else:
        values = [tf.cast(tf.range(d), dtype=heatmaps.dtype)
                  for d in tf.unstack(spatial_dims)]

    # The output of linear_expectation is likely (y, x), so we reverse it to get (x, y)
    coords = linear_expectation(heatmaps, values)
    # Reverse the last dimension to get (x, y, ...) order
    coords = tf.reverse(coords, axis=[-1])
    return coords


def dsnt(heatmaps, normalized_coordinates=True, method=None):
    """
    Differentiable spatial to numerical transform.

    This is a wrapper for soft_argmax that can optionally apply a normalization
    method to the input heatmaps.

    Args:
        heatmaps (tf.Tensor): Spatial representation of locations.
        normalized_coordinates (bool): Passed to soft_argmax. If True, returns
            coordinates in the [-1, 1] range.
        method (str, optional): If 'softmax', a `flat_softmax` is applied to
            the heatmaps before coordinate calculation. This is useful if the
            input is logits. Defaults to None.

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """
    # If the user specifies the softmax method, apply it to the heatmaps (logits).
    if method == 'softmax':
        heatmaps = flat_softmax(heatmaps)
    elif method is not None:
        raise ValueError(f"Unknown method for dsnt: '{method}'")

    # Pass the (now normalized) heatmaps and other arguments to soft_argmax.
    return soft_argmax(heatmaps, normalized_coordinates=normalized_coordinates)


def sharpen_heatmaps(heatmaps, alpha):
    """Sharpen heatmaps by increasing the contrast between high and low probabilities.

    Example:
        Approximate the mode of heatmaps using the approach described by Equation 1 of
        "FlowCap: 2D Human Pose from Optical Flow" by Romero et al.)::

            coords = soft_argmax(sharpen_heatmaps(heatmaps, alpha=6))

    Args:
        heatmaps (tf.Tensor): Heatmaps generated by the model.
        alpha (float): Sharpness factor. When ``alpha == 1``, the heatmaps will be unchanged. Use
        ``alpha > 1`` to actually sharpen the heatmaps.

    Returns:
        The sharpened heatmaps.
    """
    sharpened_heatmaps = heatmaps ** alpha

    # Normalize along spatial dimensions
    spatial_axes = list(range(2, tf.rank(sharpened_heatmaps)))
    normalizer = tf.reduce_sum(sharpened_heatmaps, axis=spatial_axes, keepdims=True)
    return sharpened_heatmaps / (normalizer + 1e-24)


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""
    orig_shape = tf.shape(inp)

    # Reshape to (B * C, H * W * ...)
    flat = tf.reshape(inp, [-1, tf.reduce_prod(orig_shape[2:])])
    flat = tf.nn.softmax(flat, axis=-1)

    # Reshape back to original shape
    return tf.reshape(flat, orig_shape)


def euclidean_losses(actual, target):
    """Calculate the Euclidean losses for multi-point samples.

    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (tf.Tensor): Predictions (B x L x D)
        target (tf.Tensor): Ground truth target (B x L x D)

    Returns:
        tf.Tensor: Losses (B x L)
    """
    tf.Assert(tf.reduce_all(tf.shape(actual) == tf.shape(target)), ['input tensors must have the same size'])
    return tf.norm(actual - target, ord='euclidean', axis=-1)


def l1_losses(actual, target):
    """Calculate the average L1 losses for multi-point samples.

    Args:
        actual (tf.Tensor): Predictions (B x L x D)
        target (tf.Tensor): Ground truth target (B x L x D)

    Returns:
        tf.Tensor: Losses (B x L)
    """
    tf.Assert(tf.reduce_all(tf.shape(actual) == tf.shape(target)), ['input tensors must have the same size'])
    return tf.reduce_mean(tf.abs(actual - target), axis=-1)


def mse_losses(actual, target):
    """Calculate the average squared L2 losses for multi-point samples.

    Args:
        actual (tf.Tensor): Predictions (B x L x D)
        target (tf.Tensor): Ground truth target (B x L x D)

    Returns:
        tf.Tensor: Losses (B x L)
    """
    tf.Assert(tf.reduce_all(tf.shape(actual) == tf.shape(target)), ['input tensors must have the same size'])
    return tf.reduce_mean(tf.square(actual - target), axis=-1)


def make_gauss(means, size, sigma, normalize=True):
    """
    Draw Gaussians. This version is robust for graph-mode execution.
    """
    # Get the number of spatial dimensions from the 'means' tensor's static shape.
    # The `tf.ensure_shape` in the calling loss function guarantees this is a known integer.
    ndims = means.shape[-1]
    if ndims is None:
        raise ValueError("The last dimension of the 'means' tensor must be statically known. "
                         "Use tf.ensure_shape in your loss function.")

    # `tf.unstack` cannot infer the number of elements from `size` if its static
    # shape is (None,). We must explicitly provide the number of dimensions.
    reversed_size_unstacked = tf.unstack(tf.reverse(size, axis=[0]), num=ndims)

    coords_list = [normalized_linspace(s, dtype=means.dtype)
                   for s in reversed_size_unstacked]

    split_means = tf.split(means, num_or_size_splits=ndims, axis=-1)

    reversed_size_float = tf.cast(tf.reverse(size, axis=[0]), dtype=means.dtype)
    stddevs_norm = 2 * sigma / reversed_size_float
    ks = [-0.5 * tf.square(1 / stddev) for stddev in tf.unstack(stddevs_norm, num=ndims)]

    reshaped_exps = []
    for i, (coords, mean, k) in enumerate(zip(coords_list, split_means, ks)):
        dist = tf.square(coords - mean)
        exp_component = tf.exp(dist * k)

        reshaped = exp_component
        for _ in range(ndims - 1 - i):
            reshaped = tf.expand_dims(reshaped, 2)
        for _ in range(i):
            reshaped = tf.expand_dims(reshaped, -1)
        reshaped_exps.append(reshaped)

    gauss = reduce(mul, reshaped_exps)

    if not normalize:
        return gauss

    spatial_axes = list(range(2, 2 + ndims))
    val_sum = tf.reduce_sum(gauss, axis=spatial_axes, keepdims=True) + 1e-24
    return gauss / val_sum


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (tf.Tensor): Predictions (B x L)
        mask (tf.Tensor, optional): Mask of points to include in the loss calculation
            (B x L), defaults to including everything
    """
    if mask is not None:
        tf.Assert(tf.reduce_all(tf.shape(losses) == tf.shape(mask)), ['mask must be the same size as losses'])
        losses = losses * tf.cast(mask, losses.dtype)
        denom = tf.reduce_sum(tf.cast(mask, losses.dtype))
    else:
        denom = tf.cast(tf.size(losses), losses.dtype)

    # Prevent division by zero
    denom = tf.maximum(denom, 1.0)

    return tf.reduce_sum(losses) / denom


def _kl(p, q, ndims):
    # Use a more robust epsilon to prevent log(0)
    eps = 1e-7
    unsummed_kl = p * (tf.math.log(p + eps) - tf.math.log(q + eps))

    rank = tf.rank(unsummed_kl)
    spatial_axes = tf.range(rank - ndims, rank)

    kl_values = tf.reduce_sum(unsummed_kl, axis=spatial_axes)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence_fn):
    ndims = tf.shape(mu_t)[-1]
    tf.print("Heatmap shape:", tf.shape(heatmaps), "mu_t shape:", tf.shape(mu_t), "sigma_t:", sigma_t, "ndims:", ndims)
    tf.Assert(tf.rank(heatmaps) == ndims + 2, ['expected heatmaps to be a {}D tensor'.format(ndims + 2)])
    tf.Assert(tf.reduce_all(tf.shape(heatmaps)[:-ndims] == tf.shape(mu_t)[:-1]),
              ['heatmap and mu_t shapes incompatible'])
    gauss = make_gauss(mu_t, tf.shape(heatmaps)[2:], sigma_t)
    divergences = divergence_fn(heatmaps, gauss, ndims)
    return divergences


def kl_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Kullback-Leibler divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (tf.Tensor): Heatmaps generated by the model.
        mu_t (tf.Tensor): Centers of the target Gaussians (in normalized units).
        sigma_t (float): Standard deviation of the target Gaussians (in pixels).

    Returns:
        Per-location KL divergences.
    """
    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _kl)


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Jensen-Shannon divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (tf.Tensor): Heatmaps generated by the model.
        mu_t (tf.Tensor): Centers of the target Gaussians (in normalized units).
        sigma_t (float): Standard deviation of the target Gaussians (in pixels).

    Returns:
        Per-location JS divergences.
    """
    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js)


def variance_reg_losses(heatmaps, sigma_t):
    """Calculate the loss between heatmap variances and target variance.

    Args:
        heatmaps (tf.Tensor): Heatmaps generated by the model (B, C, H, W).
        sigma_t (float): Target standard deviation (in pixels).

    Returns:
        Per-location sum of square errors for variance.
    """
    # mu = E[X]
    mu = soft_argmax(heatmaps, normalized_coordinates=True)  # (B, C, 2) for (x,y)

    # Get coordinate vectors
    h, w = tf.shape(heatmaps)[2], tf.shape(heatmaps)[3]
    coords_x = normalized_linspace(w, dtype=heatmaps.dtype)  # (W,)
    coords_y = normalized_linspace(h, dtype=heatmaps.dtype)  # (H,)

    # Split mu into x and y components
    mu_x, mu_y = mu[..., 0:1], mu[..., 1:2]  # (B, C, 1)

    # Calculate marginal probabilities
    marg_x = tf.reduce_sum(heatmaps, axis=2)  # (B, C, W)
    marg_y = tf.reduce_sum(heatmaps, axis=3)  # (B, C, H)

    # var = E[(X - mu)^2]
    var_x = tf.reduce_sum(marg_x * tf.square(coords_x - mu_x), axis=-1)
    var_y = tf.reduce_sum(marg_y * tf.square(coords_y - mu_y), axis=-1)
    var = tf.stack([var_x, var_y], axis=-1)  # (B, C, 2)

    # Convert variance from normalized units to pixel units
    heatmap_size = tf.cast(tf.stack([w, h]), dtype=var.dtype)
    actual_variance = var * tf.square(heatmap_size / 2.0)

    # Compare to target variance
    target_variance = sigma_t ** 2
    sq_error = tf.square(actual_variance - target_variance)

    return tf.reduce_sum(sq_error, axis=-1)


def normalized_to_pixel_coordinates(coords, size):
    """
    Convert from normalized coordinates to pixel coordinates.

    Args:
        coords (tf.Tensor): Coordinate tensor, where elements in the last dimension
            are ordered as (x, y, ...).
        size (tf.Tensor): A 1D symbolic tensor of pixel dimensions, ordered as
            (..., height, width).

    Returns:
        `coords` in pixel coordinates.
    """
    # `size` is [H, W]. Coords are (x,y). We need a size tensor of [W, H].
    # We use tf.reverse instead of Python's reversed() to handle symbolic tensors.
    size_tensor = tf.cast(tf.reverse(size, axis=[0]), dtype=coords.dtype)

    return 0.5 * ((coords + 1) * size_tensor - 1)


def pixel_to_normalized_coordinates(coords, size):
    """
    Convert from pixel coordinates to normalized coordinates.

    Args:
        coords (tf.Tensor): Coordinate tensor, where elements in the last dimension
            are ordered as (x, y, ...).
        size (tf.Tensor): A 1D symbolic tensor of pixel dimensions, ordered as
            (..., height, width).

    Returns:
        `coords` in normalized coordinates.
    """
    # This function had the same bug. We apply the same fix.
    size_tensor = tf.cast(tf.reverse(size, axis=[0]), dtype=coords.dtype)

    return ((2 * coords + 1) / size_tensor) - 1
