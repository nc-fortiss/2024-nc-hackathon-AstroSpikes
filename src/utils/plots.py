import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


def project_points(q, r, K):
    points = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)

    rotV = np.expand_dims(Rotation.from_quat(q).as_rotvec(), axis=1)
    image_points, _ = cv2.projectPoints(points, rotV, r, K, distCoeffs=np.zeros(5))

    # Draw the axes on the image
    origin = tuple(map(int, image_points[0].ravel()))
    x_axis = tuple(map(int, image_points[1].ravel()))
    y_axis = tuple(map(int, image_points[2].ravel()))
    z_axis = tuple(map(int, image_points[3].ravel()))
    return origin, x_axis, y_axis, z_axis


def visualize_both(img_list, q1_list, r1_list, q2_list, r2_list, K):
    fig, ax = plt.subplots()

    index = 0  # Start from the first image

    def on_key(event):
        """Handle spacebar press to move to the next image."""
        nonlocal index
        if event.key == ' ' and index < len(img_list) - 1:
            index += 1
            update_plot(index)  # Update with the next image
        elif event.key == 'q':  # Press 'q' to quit
            plt.close(fig)

    def update_plot(idx):
        ax.clear()  # Clear previous plots
        img = img_list[idx]
        origin, x_axis, y_axis, z_axis = project_points(q1_list[idx], r1_list[idx], K)

        cv2.line(img, origin, x_axis, (255, 0, 0), 4)  # X-axis (red)
        cv2.line(img, origin, y_axis, (0, 255, 0), 4)  # Y-axis (green)
        cv2.line(img, origin, z_axis, (0, 0, 255), 4)  # Z-axis (blue)

        origin, x_axis, y_axis, z_axis = project_points(q2_list[idx], r2_list[idx], K)

        cv2.line(img, origin, x_axis, (255, 105, 180), 4)  # X-axis (Pink)
        cv2.line(img, origin, y_axis, (144, 238, 144), 4)  # Y-axis (Light Green)
        cv2.line(img, origin, z_axis, (0, 255, 255), 4)  # Z-axis (Cyan)

        # Show the image
        ax.imshow(img)
        plt.draw()

    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot(index)  # Show the first image
    plt.show()  # Keep the plot open until manually closed
