import glob
import os
import cv2
import numpy as np
import pandas as pd
from random import randint
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

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
    
    # Extract and convert camera intrinsic matrix from the dictionary format in K
    K = np.array(K.item()['cameraMatrix'])  # Extract 'cameraMatrix' and convert to numpy array
    
    # Project points from 3D to 2D image plane
    points_image_plane = K.dot(points_camera_frame)
    x, y = (points_image_plane[0], points_image_plane[1])  # Extract x and y coordinates
    print("x,y: ", x, y)  # Debugging: print the projected points
    return x, y

def visualize(img, q, r, K, ax=None):
    """ Visualizing image with ground truth pose and projected axes. """
    if ax is None:
        ax = plt.gca()  # Get current axis if not provided

    ax.imshow(img)  # Display the image on the axis
    xa, ya = project(q, r, K)  # Project axes onto image
    
    # Define the origin point and the endpoints of the axes for drawing
    scale = 150
    c = np.array([[xa[0]], [ya[0]]])  # Center of the coordinate system in the image
    p = np.array([[xa[1], xa[2], xa[3]], [ya[1], ya[2], ya[3]]])  # Endpoints for x, y, z axes
    v = p - c  # Vectors from origin to each axis endpoint
    v = scale * v / np.linalg.norm(v)  # Normalize and scale for visualization

    # Draw arrows for x, y, z axes in red, green, blue
    ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color='r')
    ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color='g')
    ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color='b')
    return

if __name__ == '__main__':
    # Define dataset root directory
    home = os.path.expanduser("~")  # Get home directory
    root_dir = '/Users/jost/Jost/Code/2024-nc-hackathon-spades/datasets/spark'  # Dataset path
    dataset_dir = os.path.join(home, root_dir)  # Full dataset directory path

    # Get list of subdirectories in 'images' folder
    dir_list = glob.glob(os.path.join(dataset_dir, "images", "*", ""), recursive=True)
    
    # Select a random trajectory directory
    traj_dir = os.path.basename(os.path.dirname(dir_list[randint(0, len(dir_list) - 1)]))
    processed_file_path = os.path.join(dataset_dir, "train.csv")  # Path to CSV with pose data

    # Read camera intrinsic matrix from K.txt
    with open(os.path.join(dataset_dir, 'K.txt'), 'r') as file:
        array_string = file.read()  # Read file contents

    # Convert the string representation of the array to a numpy array
    K = np.array(eval(array_string))

    # Read the CSV file containing image and pose data
    data = pd.read_csv(processed_file_path)
    image_path = os.path.join(dataset_dir, 'images', traj_dir)  # Path to images in selected trajectory
    files = [os.path.basename(x) for x in glob.glob(os.path.join(image_path, '*.png'))]  # Get list of image files

    # Define grid for displaying images with projected axes
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))  # Create a 4x4 grid of subplots
    for i in range(rows):
        for j in range(cols):
            k = randint(0, 300)  # Select a random index
            image_id = files[k]  # Get image filename
            print(image_id)  # Print filename for debugging
            img_path = os.path.join(image_path, files[k])  # Full path to the image
            im_read = cv2.imread(img_path)  # Read image with OpenCV
            image = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying

            # Get pose data (translation and quaternion) for the current image
            i_data = data.loc[data['filename'] == image_id]  # Filter data for the selected image
            Tx = i_data['Tx'].values.squeeze()  # Translation x-coordinate
            Ty = i_data['Ty'].values.squeeze()  # Translation y-coordinate
            Tz = i_data['Tz'].values.squeeze()  # Translation z-coordinate
            Qx = i_data['Qx'].values.squeeze()  # Quaternion x-component
            Qy = i_data['Qy'].values.squeeze()  # Quaternion y-component
            Qz = i_data['Qz'].values.squeeze()  # Quaternion z-component
            Qw = i_data['Qw'].values.squeeze()  # Quaternion w-component

            # Create translation and rotation vectors
            r = np.array([Tx, Ty, Tz])  # Translation vector
            q = np.array([Qx, Qy, Qz, Qw])  # Quaternion representing orientation
            print(r, q)  # Print translation and quaternion for debugging

            # Visualize the image with projected axes
            visualize(image, q, r, K, ax=axes[i][j])
            axes[i][j].axis('off')  # Hide axis for cleaner display
    fig.tight_layout()  # Adjust layout for better spacing
    plt.show()  # Display the figure