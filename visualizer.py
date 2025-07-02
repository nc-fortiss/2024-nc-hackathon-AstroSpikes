import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Visualizer:
    def __init__(self, dest:str):
        self.dest = dest
        os.makedirs(self.dest, exist_ok=True)

    def visualize(self, keypoints, predictions, img_path, name = None):
        """
        Visualizes keypoints and predictions on the image.

        Args:
            keypoints (np.ndarray): Array of shape (N, 2) for keypoints.
            predictions (np.ndarray): Array of shape (M, 2) for predictions.
            img (np.ndarray): Image array of shape (H, W, C).
        """
        img = plt.imread(img_path)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        # Plot keypoints
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c='blue', label='Keypoints', s=50)

        # Plot predictions
        plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predictions', s=50)

        plt.legend()
        plt.axis('on')

        # Save the figure
        file_path = os.path.join(self.dest, f"{name}.png" if name else img_path.split('/')[-1].replace('.jpg', '.png'))
        plt.savefig(file_path)
        plt.close()

if __name__ == "__main__":
    img_path = "/Users/jost/Jost/Code/2024-nc-hackathon-spades/generated_dataset/lnes/RT000/img000_RT000.png"
    visualizer = Visualizer(dest="/Users/jost/Jost/Code/2024-nc-hackathon-spades/visualizations/keypoints")
    json_path = "/Users/jost/Jost/Code/2024-nc-hackathon-spades/spades_label/train.json"

    df = pd.read_json(json_path)
    row = df[df["filename"]=="img000_RT000.png"].iloc[0]
    keypoints = row["keypoints"]
    # make every two values the coordinates of a point, then skip the third one
    keypoints = np.array(keypoints).reshape(-1, 3)[:, :2]
    predictions = keypoints + np.random.normal(0,10,size=keypoints.shape)
    visualizer.visualize(keypoints, predictions, img_path, name="img000_RT000")