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
        ax.cla()  # Clear axis for new plots

    ax.imshow(img)  # Display the image on the axis
    xa, ya = project(q, r, K)  # Project axes onto image
    
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

# def compare_pred(n_traj, n_img):
#     dataset_root_dir = "/Users/jost/Jost/Code/2024-nc-hackathon-spades"

def get_model_prediction(img, model_path):
    """ Get model prediction for the input image """
    tf.keras.utils.get_custom_objects().update({'PoseEstimationLoss': PoseEstimationLoss})

    img = tf.expand_dims(img, axis = 0)
    print(img.shape)
    # Load the model from the specified path
    model = tf.keras.models.load_model(model_path)
    model.summary()  # Display model summary for debugging
    [[x,y,z,qx,qy,qz,qw]] = model.predict(img)  # Get model predictions for the input image
    return np.array([x,y,z]), np.array([qx,qy,qz,qw])  # Return the predicted translation and quaternion

def save_visualization(img_name, model_path, K, ax = None):
    img_path = os.path.join("/Users/jost/Jost/Code/2024-nc-hackathon-spades/datasets/spark/images/traj1", img_name)
    img = cv2.imread(img_path)  # Read the input image

    #create folder visualizations if not there
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")

    #Predictions
    r, q = get_model_prediction(img, model_path)
    print("Predicted: ", r, q)  # Print the predicted translation and quaternion for debugging

    #Labels
    df = pd.read_csv("/Users/jost/Jost/Code/2024-nc-hackathon-spades/datasets/spark/train.csv")
    label = df.loc[df['filename'] == img_name]
    x = label.iloc[0]['Tx']
    y = label.iloc[0]['Ty']
    z = label.iloc[0]['Tz']
    qx = label.iloc[0]['Qx']
    qy = label.iloc[0]['Qy']
    qz = label.iloc[0]['Qz']
    qw = label.iloc[0]['Qw']

    r1 = np.array([x,y,z])
    q1 = np.array([qx,qy,qz,qw])
    print("Label: ", r1, q1)  # Print the ground truth translation and quaternion for debuggingp

    visualize(img, q, r, K)  # Visualize the image with projected axes
    plt.savefig("visualizations/" + ("predicted" + img_name))  # Save the figure to the 'visualizations' folder
    plt.close()  # Close the figure to free up memory


    visualize(img, q1, r1, K)  # Visualize the image with projected axes
    plt.savefig("visualizations/" + ("label" + img_name))  # Save the figure to the 'visualizations' folder
    plt.close()  # Close the figure to free up memory

if __name__ == '__main__':

    # Read camera intrinsic matrix from K.txt
    with open("/Users/jost/Jost/Code/2024-nc-hackathon-spades/datasets/spark/K.txt", 'r') as file:
        array_string = file.read()  # Read file contents

    # Convert the string representation of the array to a numpy array
    K = np.array(eval(array_string))
    
    #loop through all pngs that are avilable
    #get all pngs of folder


    img_name = "img002_RT002.png"
    if img_name.endswith(".png"):
        print(img_name)
        save_visualization(img_name, model_path="/Users/jost/Jost/Code/2024-nc-hackathon-spades/DataVisualization/latest_model_3231.keras", K=K)
    
    # # Define dataset root directory
    # home = os.path.expanduser("~")  # Get home directory
    # root_dir = '/Users/jost/Jost/Code/2024-nc-hackathon-spades/datasets/spark'  # Dataset path
    # dataset_dir = os.path.join(home, root_dir)  # Full dataset directory path

    # # Get list of subdirectories in 'images' folder
    # dir_list = glob.glob(os.path.join(dataset_dir, "images", "*", ""), recursive=True)
    
    # # Select a random trajectory directory
    # traj_dir = os.path.basename(os.path.dirname(dir_list[randint(0, len(dir_list) - 1)]))
    # processed_file_path = os.path.join(dataset_dir, "train.csv")  # Path to CSV with pose data

    # # Read camera intrinsic matrix from K.txt
    # with open(os.path.join(dataset_dir, 'K.txt'), 'r') as file:
    #     array_string = file.read()  # Read file contents

    # # Convert the string representation of the array to a numpy array
    # K = np.array(eval(array_string))

    # # Read the CSV file containing image and pose data
    # data = pd.read_csv(processed_file_path)
    # image_path = os.path.join(dataset_dir, 'images', traj_dir)  # Path to images in selected trajectory
    # files = [os.path.basename(x) for x in glob.glob(os.path.join(image_path, '*.png'))]  # Get list of image files

    # # Define grid for displaying images with projected axes
    # rows, cols = 4, 4
    # fig, axes = plt.subplots(rows, cols, figsize=(20, 20))  # Create a 4x4 grid of subplots
    # for i in range(rows):
    #     for j in range(cols):
    #         k = randint(0, 300)  # Select a random index
    #         image_id = files[k]  # Get image filename
    #         print(image_id)  # Print filename for debugging
    #         img_path = os.path.join(image_path, files[k])  # Full path to the image
    #         im_read = cv2.imread(img_path)  # Read image with OpenCV
    #         image = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying

    #         # Get pose data (translation and quaternion) for the current image
    #         i_data = data.loc[data['filename'] == image_id]  # Filter data for the selected image
    #         Tx = i_data['Tx'].values.squeeze()  # Translation x-coordinate
    #         Ty = i_data['Ty'].values.squeeze()  # Translation y-coordinate
    #         Tz = i_data['Tz'].values.squeeze()  # Translation z-coordinate
    #         Qx = i_data['Qx'].values.squeeze()  # Quaternion x-component
    #         Qy = i_data['Qy'].values.squeeze()  # Quaternion y-component
    #         Qz = i_data['Qz'].values.squeeze()  # Quaternion z-component
    #         Qw = i_data['Qw'].values.squeeze()  # Quaternion w-component

    #         # Create translation and rotation vectors
    #         r = np.array([Tx, Ty, Tz])  # Translation vector
    #         q = np.array([Qx, Qy, Qz, Qw])  # Quaternion representing orientation
    #         print(r, q)  # Print translation and quaternion for debugging

    #         # Visualize the image with projected axes
    #         visualize(image, q, r, K, ax=axes[i][j])
    #         axes[i][j].axis('off')  # Hide axis for cleaner display
    # fig.tight_layout()  # Adjust layout for better spacing
    # plt.show()  # Display the figure