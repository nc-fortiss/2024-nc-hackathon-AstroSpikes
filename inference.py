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
        pred_pos, pred_quat = self.model.predict(img)
        pred_quat = tf.linalg.normalize(pred_quat, axis=-1)[0]
        return np.array(pred_pos).squeeze(), np.array(pred_quat).squeeze()


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
    data_root = str(os.path.join(config.root.dataset, config.data.source, config.data.transformation))

    with open(config.root.test_data, "r") as file:
        reader = csv.reader(file)
        test_data = list(reader)

    for file_name in test_data:
        # print(file_name)
        index = 0
        img_list = []
        gt_q_list = []
        gt_r_list = []
        pred_q_list = []
        pred_r_list = []

        seq = file_name[0].split('.')[0]
        df = pd.read_csv(os.path.join(config.root.data_out, 'labels', file_name[0]))
        for img_path in sorted(glob.glob(os.path.join(data_root, seq, '*.png'))):
            img_id = os.path.basename(img_path)
            # print(img_path)
            pred_r, pred_q = infer.get_model_prediction(img_path)
            data = df.loc[df['filename'] == img_id]
            gt_r = np.array(data[['Tx', 'Ty', 'Tz']]).squeeze()
            gt_q = np.array(data[['Qx', 'Qy', 'Qz', 'Qw']]).squeeze()
            img = cv2.imread(img_path)
            img_list.append(img)
            gt_q_list.append(gt_q)
            gt_r_list.append(gt_r)
            pred_q_list.append(pred_q)
            pred_r_list.append(pred_r)
            index += 1
            if index % 100 == 0:
                visualize_both(img_list, gt_q_list, gt_r_list, pred_q_list, pred_r_list, K)
                index = 0
                img_list = []
                gt_q_list = []
                gt_r_list = []
                pred_q_list = []
                pred_r_list = []
