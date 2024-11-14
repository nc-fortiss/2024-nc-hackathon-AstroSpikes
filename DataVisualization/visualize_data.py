import glob
import os
import cv2
import numpy as np
import pandas as pd
from random import randint
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tensorflow import keras
import tensorflow as tf
from losses import PoseEstimationLoss
import tonic


class Visualizer():
    def __init__(self, root_frames_dir, model_path, K_path, dest_dir):
        self.root_frames_dir = root_frames_dir
        self.model_path = model_path
        self.K_path = K_path
        self.dest_dir = dest_dir

        self.root_frames_dir = root_frames_dir

        self.model_path = model_path
        tf.keras.utils.get_custom_objects().update({'PoseEstimationLoss': PoseEstimationLoss})
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.summary()
        print("Model loaded successfully!")

        self.K_path = K_path
        with open(K_path, 'r') as file:
            array_string = file.read()
        self.K = np.array(eval(array_string))
        self.K = np.array(self.K.item()['cameraMatrix'])
        print("Camera intrinsic matrix K loaded successfully!")

        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        print("Destination directory created successfully!")

    def project(self, q, r):
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
        points_image_plane = self.K.dot(points_camera_frame)
        x, y = (points_image_plane[0], points_image_plane[1])  # Extract x and y coordinates
        # print("x,y: ", x, y)  # Debugging: print the projected points
        return x, y

    def visualize(self, img, q, r, ax=None):
        """ Visualizing image with ground truth pose and projected axes. """
        if ax is None:
            ax = plt.gca()  # Get current axis if not provided
            ax.cla()  # Clear axis for new plots

        ax.imshow(img)  # Display the image on the axis
        xa, ya = project(q, r, self.K)  # Project axes onto image
        
        # Define the origin point and the endpoints of the axes for drawing
        scale = 50
        c = np.array([[xa[0]], [ya[0]]])  # Center of the coordinate system in the image
        #convert c
        c = c - np.array([[280], [0]])
        c = c/3

        p = np.array([[xa[1], xa[2], xa[3]], [ya[1], ya[2], ya[3]]])  # Endpoints for x, y, z axes
        #convert p
        p = p - np.array([[280], [0]])
        p = p/3

        v = p - c  # Vectors from origin to each axis endpoint
        v = scale * v / np.linalg.norm(v)  # Normalize and scale for visualization

        # Draw arrows for x, y, z axes in red, green, blue
        ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color='r')
        ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color='g')
        ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color='b')
        return

    def visualize_both(self, img, q1, r1, q2, r2, ax=None):
        #plot both the predicted and the label
        if ax is None:
            ax = plt.gca()
            ax.cla()
        
        ax.imshow(img)
        scale = 50

        xa1, ya1 = self.project(q1, r1)
        xa2, ya2 = self.project(q2, r2)
        
        c1, p1 = self.transform_points(xa1, ya1)
        c2, p2 = self.transform_points(xa2, ya2)

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

    def transform_points(self, xa, ya):
        #transform the points to the center of the image
        c = np.array([[xa[0]], [ya[0]]])
        c = c - np.array([[280], [0]])
        c = c/3

        p = np.array([[xa[1], xa[2], xa[3]], [ya[1], ya[2], ya[3]]])
        p = p - np.array([[280], [0]])
        p = p/3
        return c, p

    def get_model_prediction(self, img):
        """ Get model prediction for the input image """
        img = tf.expand_dims(img, axis = 0)
        # print(img.shape)
        
        [[x,y,z,qx,qy,qz,qw]] = self.model.predict(img)  # Get model predictions for the input image
        return np.array([x,y,z]), np.array([qx,qy,qz,qw])  # Return the predicted translation and quaternion

    def create_graphic(self, img_name, ax = None):
        n_traj = img_name[-7:-4]
        img_path = os.path.join(self.root_frames_dir, ("seq_RT" + n_traj), img_name)
        #check if img_path is correct

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

        r_label = np.array([x,y,z])
        q_label = np.array([qx,qy,qz,qw])
        # print("Label: ", r, q)

        self.visualize_both(img, q_label, r_label, q_pred, r_pred, ax)
        plt.savefig(os.path.join(self.dest_dir, img_name))
        plt.close()


# if __name__ == '__main__':
#     root_frames_dir = "/Users/jost/Jost/Code/2024-nc-hackathon-spades/frames"
#     model_path = "/Users/jost/Jost/Code/2024-nc-hackathon-spades/DataVisualization/latest_model_3231.keras"
#     K_path = "/Users/jost/Jost/Code/2024-nc-hackathon-spades/DataVisualization/camera.json"
#     dest_dir = "/Users/jost/Jost/Code/2024-nc-hackathon-spades/visualizations"

#     visualizer = Visualizer(root_frames_dir, model_path, K_path, dest_dir)
#     visualizer.create_graphic("img002_RT002.png")
#     print("finished")

