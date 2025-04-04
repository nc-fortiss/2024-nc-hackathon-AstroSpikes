import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf, DictConfig
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from src.losses.poseloss import PoseEstimationLoss
from src.models.mobilenet import MobilenetModel
from src.utils.plots import visualize_both


class PoseInference:
    def __init__(self, cfg: DictConfig):
        print(cfg.model.trained_weights)
        tf.keras.utils.get_custom_objects().update({'PoseEstimationLoss': PoseEstimationLoss})
        self.input_image_size = cfg.data.input_size
        input_shape = list(cfg.data.input_size)  # Convert ListConfig to a standard list
        self.model = MobilenetModel(input_size=input_shape, pretrained=cfg.model.pretrained)
        self.model.build(input_shape=(None, *input_shape))  # None is for batch size
        self.model.load_weights(cfg.model.trained_weights)

        self.model.summary()
        print("Model loaded successfully!")

        with open(os.path.join(cfg.root.dataset, 'camera.json'), "r") as jfile:
            data = json.load(jfile)
        self.K = np.array(data["cameraMatrix"])
        print("Camera intrinsic matrix K loaded successfully!")

    def _load_image(self, filepath):
        """Loads and preprocesses a single image."""
        image = load_img(filepath, target_size=self.input_image_size)
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]
        image = tf.expand_dims(image, axis=0)
        return image

    def get_model_prediction(self, filepath):
        """ Get model prediction for the input image """
        img = self._load_image(filepath)
        pos, rot = self.model.predict(img)
        pred_quat = tf.linalg.normalize(rot, axis=-1)[0]
        return np.array(pos).squeeze(), np.array(pred_quat).squeeze()


if __name__ == '__main__':

    K = np.array([
        [1258.6057531097028, 0.0, 640],
        [0.0, 1258.6057531097028, 360],
        [0.0, 0.0, 1.0]
    ])
    config_path = "configs/mobilenet.yaml"
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    infer = PoseInference(cfg=config)

    df = pd.read_csv(config.root.test_data)

    img_list = []
    gt_q_list = []
    gt_r_list = []
    pred_q_list = []
    pred_r_list = []

    results = []

    for index, row in df.iterrows():
        pred_r, pred_q = infer.get_model_prediction(row['filepath'])
        results.append([
            row['filepath'],
            pred_r[0], pred_r[1], pred_r[2],
            pred_q[0], pred_q[1], pred_q[2], pred_q[3]
        ])

        gt_r = np.array(row[['Tx', 'Ty', 'Tz']], dtype=np.float64)
        gt_q = np.array(row[['Qx', 'Qy', 'Qz', 'Qw']], dtype=np.float64)
        # print(f"\nTranslation-Pred: {pred_r}, GT: {gt_r}")
        # print(f"\nRotation-Pred: {pred_q}, GT: {gt_q}")
        img = cv2.imread(row['filepath'])
        img_list.append(img)
        gt_q_list.append(gt_q)
        gt_r_list.append(gt_r)
        pred_q_list.append(pred_q)
        pred_r_list.append(pred_r)
        if index > 1 and index % 100 == 0:
            # print(gt_r_list)
            visualize_both(img_list, gt_q_list, gt_r_list, pred_q_list, pred_r_list, K)
            img_list = []
            gt_q_list = []
            gt_r_list = []
            pred_q_list = []
            pred_r_list = []

    # Convert to a DataFrame
    pred_df = pd.DataFrame(results, columns=[
        'filepath', 'Tx', 'Ty', 'Tz', 'Qx', 'Qy', 'Qz', 'Qw'
    ])
    # Write to CSV
    pred_df.to_csv("predictions.csv", index=False)
