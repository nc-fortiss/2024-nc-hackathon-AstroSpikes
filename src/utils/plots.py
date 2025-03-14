from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import os
import pandas as pd


def transform_points(xa, ya):
    # transform the points to the center of the image
    c = np.array([[xa[0]], [ya[0]]])
    c = c - np.array([[280], [0]])
    c = c / 3

    p = np.array([[xa[1], xa[2], xa[3]], [ya[1], ya[2], ya[3]]])
    p = p - np.array([[280], [0]])
    p = p / 3
    return c, p


def project(q, r, K):
    """ Projecting points to image frame to draw axes """
    # Reference points in satellite frame for drawing axes (origin and unit vectors along x, y, z)
    p_axes = np.array([[0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)  # Transpose for easier matrix operations

    # Transformation to camera frame using rotation matrix and translation vector
    pose_mat = np.hstack((Rotation.from_quat(q).as_matrix(), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)  # Transform points to camera frame

    # Normalize by z-coordinate to get homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # Project points from 3D to 2D image plane
    points_image_plane = K.dot(points_camera_frame)
    x, y = (points_image_plane[0], points_image_plane[1])  # Extract x and y coordinates
    # print("x,y: ", x, y)  # Debugging: print the projected points
    return x, y


def visualize_both(img, q1, r1, q2, r2, K, ax=None):
    # plot both the predicted and the label
    if ax is None:
        ax = plt.gca()
        ax.cla()

    ax.imshow(img)
    scale = 50

    xa1, ya1 = project(q1, r1, K)
    xa2, ya2 = project(q2, r2, K)

    c1, p1 = transform_points(xa1, ya1)
    c2, p2 = transform_points(xa2, ya2)

    v1 = p1 - c1
    v1 = scale * v1 / np.linalg.norm(v1)

    v2 = p2 - c2
    v2 = scale * v2 / np.linalg.norm(v2)

    # Darker shades for the label
    ax.arrow(c1[0, 0], c1[1, 0], v1[0, 0], v1[1, 0], head_width=10, color='#990000')  # Dark red
    ax.arrow(c1[0, 0], c1[1, 0], v1[0, 1], v1[1, 1], head_width=10, color='#009900')  # Dark green
    ax.arrow(c1[0, 0], c1[1, 0], v1[0, 2], v1[1, 2], head_width=10, color='#000099')  # Dark blue

    # Lighter shades for the prediction
    ax.arrow(c2[0, 0], c2[1, 0], v2[0, 0], v2[1, 0], head_width=10, color='#FF6666')  # Light red
    ax.arrow(c2[0, 0], c2[1, 0], v2[0, 1], v2[1, 1], head_width=10, color='#66FF66')  # Light green
    ax.arrow(c2[0, 0], c2[1, 0], v2[0, 2], v2[1, 2], head_width=10, color='#6666FF')  # Light blue
    return


def create_graphic(self, img_name, ax=None):
    n_traj = img_name[-7:-4]
    img_path = os.path.join(self.root_frames_dir, ("seq_RT" + n_traj), img_name)
    # check if img_path is correct

    # if os.path.exists(img_path):
    #     print("img path exists")
    # else:
    #     print("img path does not exist")
    img = cv2.imread(img_path)

    if not os.path.exists(self.dest_dir):
        os.makedirs(self.dest_dir)

    r_pred, q_pred = self.get_model_prediction(img)
    # print("Predicted: ", r, q)

    df = pd.read_csv(os.path.join(self.root_frames_dir, ("seq_RT" + n_traj), ("seq_RT" + n_traj + ".csv")))
    label = df.loc[df['filename'] == img_name]
    if label.empty:
        print(f"Image {img_name} not found in the DataFrame.")
        return  # Exit the function if no matching row is found

    x = label.iloc[0]['Tx']
    y = label.iloc[0]['Ty']
    z = label.iloc[0]['Tz']
    qx = label.iloc[0]['Qx']
    qy = label.iloc[0]['Qy']
    qz = label.iloc[0]['Qz']
    qw = label.iloc[0]['Qw']

    r_label = np.array([x, y, z])
    q_label = np.array([qx, qy, qz, qw])
    # print("Label: ", r, q)

    self.visualize_both(img, q_label, r_label, q_pred, r_pred, ax)
    plt.savefig(os.path.join(self.dest_dir, img_name))
    plt.close()
    return q_label, r_label, q_pred, r_pred
