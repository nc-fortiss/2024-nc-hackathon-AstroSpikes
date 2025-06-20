import os
import shutil
import time

import numpy as np
import pandas as pd
import tonic
from PIL import Image
import cv2
from omegaconf import OmegaConf

class DatasetGenerator():
    def __init__(self, frames_dir, output_dir, labels_dir):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.labels_dir = labels_dir
        self.train = None
        self.test = None
        self.val = None

    def load_json(self):
        self.train = pd.read_json(self.labels_dir + "/train.json")
        self.test = pd.read_json(self.labels_dir + "/test.json")
        self.val = pd.read_json(self.labels_dir + "/val.json")

    def get_img(self, filename):
        #remove .png at the end
        filename_removed = filename.replace(".png", "")
        number, traj = filename_removed.split("_")
        print("Trajectory:", traj)
        path_img = self.frames_dir + "/" + traj + "/" + filename
        # add traj to path and then the filename
        img = cv2.imread(path_img)
        if img is None:
            print(f"Image {filename} not found in {path_img}")
            return None
        return img

    def crop_scale_img(self, frame, keypoints, bbox):
        img_cropped = frame[int(bbox[0, 1]):int(bbox[1, 1]), int(bbox[0, 0]):int(bbox[1, 0])]
        img_cropped = cv2.resize(img_cropped, (224, 224))
        keypoints_cropped = keypoints - np.array([bbox[0, 0], bbox[0, 1]])
        keypoints_cropped = keypoints_cropped * (224 / (bbox[1, 0] - bbox[0, 0]))
        print("Cropped keypoints:", keypoints_cropped)
        return keypoints_cropped, img_cropped

    def start_generating(self):
        self.load_json()
        if self.train is None or self.test is None or self.val is None:
            print("Json files could not loaded. Please load the files first.")
            return


        # make an empty csv with the following columns:
        # filepath, k0x, k0y, ...., k7x, k7y
        columns = ['filepath'] + [f'k{i}x' for i in range(8)] + [f'k{i}y' for i in range(8)]
        train_df = pd.DataFrame(columns=columns)

        for index, row in self.train.iterrows():
            filename = row['filename']
            keypoints = np.array(row['keypoints'])
            keypoints = np.array(keypoints).reshape(-1, 3)[:, :2]
            bbox = row['bbox']
            bbox = np.array(bbox)
            bbox = bbox.reshape(-1, 2)

            img = self.get_img(filename)
            if img is None:
                continue

            keypoints_cropped, img_cropped = self.crop_scale_img(img, keypoints, bbox)

            # Save the cropped image
            output_path = self.output_dir + "/" + "train" + "/" + filename

            # save the cropped image and get a confirmation
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_cropped)
            print(f"Saved cropped image to {output_path}")

            # save the cropped keypoints in the df
            keypoints_row = [output_path] + keypoints_cropped.flatten().tolist()
            train_df.loc[len(train_df)] = keypoints_row
            print(f"Processed {filename}")

        # save the train_df to a csv file
        train_csv_path = self.output_dir + "/" + "train" + "/" + "/keypoints.csv"
        os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
        train_df.to_csv(train_csv_path, index=False)
        print(f"Train keypoints saved to {train_csv_path}")



if __name__ == "__main__":
    # Load configuration file
    try:
        cfg = OmegaConf.load("configs/mobilenet.yaml")
        print(cfg)
    except Exception as e:
        print("Error loading YAML:", e)

    # Load root directory and output directory
    frame_dir = cfg.root.dataset + '/' + cfg.data.transformation
    output_dir = cfg.root.data_out + '/' + cfg.data.transformation + "_cropped"
    labels_dir = cfg.root.labels
    os.makedirs(output_dir, exist_ok=True)

    print(f"Frame directory: {frame_dir}")
    print(f"Output directory: {output_dir}\n")

    # filepath, k0x, k0y, ...., k7x, k7y
    dg = DatasetGenerator(frame_dir, output_dir, labels_dir)
    dg.start_generating()



