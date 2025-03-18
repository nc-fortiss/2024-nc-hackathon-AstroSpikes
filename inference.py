import os

import numpy as np
import tensorflow as tf
from src.losses.poseloss import PoseEstimationLoss
from omegaconf import OmegaConf, DictConfig
import json
import csv
import cv2
from src.models.mobilenet import MobilenetModel
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array


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
        return np.array(pred_pos), np.array(pred_quat)  # Return the predicted translation and quaternion


if __name__ == '__main__':

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
        print(file_name)
        seq = file_name[0].split('.')[0]
        for img_path in glob.glob(os.path.join(data_root, seq, '*.png')):
            print(img_path)
            r_pred, q_pred = infer.get_model_prediction(img_path)
            print(r_pred, q_pred)
