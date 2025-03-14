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


class PoseInference:
    def __init__(self, cfg: DictConfig):

        print(cfg.model.trained_weights)
        tf.keras.utils.get_custom_objects().update({'PoseEstimationLoss': PoseEstimationLoss})

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

        # self.dest_dir = dest_dir
        # if not os.path.exists(self.dest_dir):
        #     os.makedirs(self.dest_dir)
        # print("Destination directory created successfully!")

    def get_model_prediction(self, img):
        """ Get model prediction for the input image """
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_with_crop_or_pad(img, target_height=224, target_width=224)

        [[x, y, z, qx, qy, qz, qw]] = self.model.predict(img)  # Get model predictions for the input image
        return np.array([x, y, z]), np.array([qx, qy, qz, qw])  # Return the predicted translation and quaternion


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
        for filename in glob.glob(os.path.join(data_root, seq, '*.png')):
            img = cv2.imread(os.path.join(data_root, seq, filename))
            r_pred, q_pred = infer.get_model_prediction(img)
            print(r_pred, q_pred)
