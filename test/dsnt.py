from src.utils.kdsnt import *


# ==============================================================================
# Test Functions
# ==============================================================================

def test_create_meshgrid():
    print("--- Running test_create_meshgrid ---")
    # Test normalized coordinates
    grid_norm = create_meshgrid_tf(2, 2, normalized_coordinates=True)
    expected_norm = tf.constant([[[[-1., -1.], [1., -1.]],
                                  [[-1., 1.], [1., 1.]]]], dtype=tf.float32)
    tf.debugging.assert_near(grid_norm, expected_norm, rtol=1e-6)
    print("  Normalized grid: PASS")

    # Test unnormalized (pixel) coordinates
    grid_pixel = create_meshgrid_tf(2, 2, normalized_coordinates=False)
    expected_pixel = tf.constant([[[[0., 0.], [1., 0.]],
                                   [[0., 1.], [1., 1.]]]], dtype=tf.float32)
    tf.debugging.assert_near(grid_pixel, expected_pixel, rtol=1e-6)
    print("  Pixel grid: PASS")
    print("--- test_create_meshgrid: All tests PASSED ---")


def test_spatial_softmax2d():
    print("\n--- Running test_spatial_softmax2d ---")
    heatmaps = tf.constant([[[
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 1., 2.]]]], dtype=tf.float32)

    output = spatial_softmax2d(heatmaps)

    # 1. Test shape: The output shape must match the input shape.
    tf.debugging.assert_equal(tf.shape(output), tf.shape(heatmaps),
                              message="Output shape does not match input shape.")
    print("  Output shape is correct: PASS")

    # 2. Test sum: The spatial probabilities for each channel must sum to 1.0.
    # Reshape to (batch, channels, H*W) and sum over the last dimension.
    output_sum = tf.reduce_sum(tf.reshape(output, [1, 1, -1]), axis=-1)
    tf.debugging.assert_near(output_sum, tf.constant([[1.0]], dtype=tf.float32), rtol=1e-6,
                             message="Probabilities do not sum to 1.")
    print("  Probabilities sum to 1.0: PASS")

    # 3. Test peak location: The argmax of the output should be at the same location as the input.
    input_flat = tf.reshape(heatmaps, [-1])
    output_flat = tf.reshape(output, [-1])
    tf.debugging.assert_equal(tf.argmax(input_flat), tf.argmax(output_flat),
                              message="Peak location (argmax) shifted after softmax.")
    print("  Peak location is preserved: PASS")

    print("--- test_spatial_softmax2d: All tests PASSED ---")


def test_spatial_expectation2d():
    print("\n--- Running test_spatial_expectation2d ---")
    # A perfect heatmap with one hot pixel at (row=2, col=1) -> (x=1, y=2)
    heatmaps = tf.constant([[[
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 1., 0.]]]], dtype=tf.float32)

    # Test with unnormalized coordinates
    output_pixel = spatial_expectation2d(heatmaps, normalized_coordinates=False)
    expected_pixel = tf.constant([[[1., 2.]]], dtype=tf.float32)
    tf.debugging.assert_near(output_pixel, expected_pixel, rtol=1e-6)
    print("  Unnormalized expectation: PASS")

    # Test with normalized coordinates. For a 3x3 grid, (x=1, y=2) maps to (0, 1)
    # x_coords: [-1, 0, 1], y_coords: [-1, 0, 1]
    output_norm = spatial_expectation2d(heatmaps, normalized_coordinates=True)
    expected_norm = tf.constant([[[0., 1.]]], dtype=tf.float32)
    tf.debugging.assert_near(output_norm, expected_norm, rtol=1e-6)
    print("  Normalized expectation: PASS")
    print("--- test_spatial_expectation2d: All tests PASSED ---")


def test_render_gaussian2d():
    print("\n--- Running test_render_gaussian2d ---")
    size = (5, 5)
    # Place the mean at the center of the normalized grid
    mean = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    std = tf.constant([[0.5, 0.5]], dtype=tf.float32)

    gauss_map = render_gaussian2d(mean, std, size, normalized_coordinates=True)

    # 1. Check shape
    expected_shape = (1, 5, 5)
    tf.debugging.assert_equal(tf.shape(gauss_map), expected_shape,
                              message="Shape mismatch")
    print(f"  Output shape {gauss_map.shape} is correct: PASS")

    # 2. Check that the distribution sums to 1
    map_sum = tf.reduce_sum(gauss_map)
    tf.debugging.assert_near(map_sum, 1.0, rtol=1e-5,
                             message="Gaussian map does not sum to 1")
    print(f"  Map sum ({map_sum:.6f}) is close to 1.0: PASS")

    # 3. Check that the peak is at the center (index H//2, W//2)
    peak_value = gauss_map[0, 2, 2]
    # Check against neighbors
    tf.debugging.assert_greater(peak_value, gauss_map[0, 2, 1])
    tf.debugging.assert_greater(peak_value, gauss_map[0, 1, 2])
    print(f"  Peak value ({peak_value:.4f}) is at the center: PASS")
    print("--- test_render_gaussian2d: All tests PASSED ---")


# Add this test function to the main testing block in your dsnt_tf.py file.

def test_normalize_pixel_coordinates():
    print("\n--- Running test_normalize_pixel_coordinates ---")
    # Use an image size with a clear center pixel
    height, width = 101, 201

    # Define keypoints at the corners and the center in pixel coordinates
    pixel_coords = tf.constant([
        [0.0, 0.0],  # Top-left corner
        [200.0, 0.0],  # Top-right corner
        [0.0, 100.0],  # Bottom-left corner
        [200.0, 100.0],  # Bottom-right corner
        [100.0, 50.0]  # Center
    ], dtype=tf.float32)

    # Define the expected normalized coordinates
    expected_norm_coords = tf.constant([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], dtype=tf.float32)

    # Perform the normalization
    normalized_coords = normalize_pixel_coordinates(pixel_coords, height, width)

    # Assert that the result is close to the expected value
    tf.debugging.assert_near(normalized_coords, expected_norm_coords, rtol=1e-6)
    print("  Normalization of corners and center is correct: PASS")
    print("--- test_normalize_pixel_coordinates: All tests PASSED ---")


# Add this test function to the main testing block in your dsnt_tf.py file.

def test_denormalize_pixel_coordinates():
    print("\n--- Running test_denormalize_pixel_coordinates ---")
    height, width = 101, 201

    # Define keypoints at the corners and the center in normalized coordinates
    norm_coords = tf.constant([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], dtype=tf.float32)

    # Define the expected pixel coordinates
    expected_pixel_coords = tf.constant([
        [0.0, 0.0],  # Top-left corner
        [200.0, 0.0],  # Top-right corner
        [0.0, 100.0],  # Bottom-left corner
        [200.0, 100.0],  # Bottom-right corner
        [100.0, 50.0]  # Center
    ], dtype=tf.float32)

    # Perform the denormalization
    pixel_coords = denormalize_pixel_coordinates(norm_coords, height, width)

    # Assert that the result is close to the expected value
    tf.debugging.assert_near(pixel_coords, expected_pixel_coords, rtol=1e-6)
    print("  Denormalization of corners and center is correct: PASS")
    print("--- test_denormalize_pixel_coordinates: All tests PASSED ---")


def main():
    """Run all test functions."""
    test_create_meshgrid()
    test_spatial_softmax2d()
    test_spatial_expectation2d()
    test_render_gaussian2d()
    test_normalize_pixel_coordinates()
    test_denormalize_pixel_coordinates()
    print("\n\n✅ All function evaluations completed successfully!")


if __name__ == "__main__":
    main()
