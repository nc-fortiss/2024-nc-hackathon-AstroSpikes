import numpy as np
import time
import cv2
import sys
import matplotlib.pyplot as plt 
import json
import logging

import os
import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
from datetime import datetime
import pandas as pd
import glob
import logging


from tensorflow.keras.callbacks import ModelCheckpoint
from src.dataloaders.spades import create_dataset
from src.losses.poseloss import geodesic_dist, position_mse_loss, ori_error, rel_l2_error, mpkpe_heatmap, mpkpe_regression
from src.losses.heatmaploss import heatmap_loss
from src.models.mobilenet_regression import MobilenetModel_regression
from src.models.mobilenet_heatmap import mobilenet_heatmap
import quantizeml
from cnn2snn import convert
import akida as ak
import cnn2snn

from akida_models.imagenet import get_preprocessed_samples, akidanet_imagenet
from akida_models.imagenet import akidanet_imagenet, mobilenet_imagenet
from akida_models import fetch_file
from tensorflow.keras.models import load_model
import keras

os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = "v1"
print("\nCONFIGURATION")
print("TensorFlow version: ", tf.__version__)
print("    MetaTF version: ", ak.__version__)
print('     Akida version: ', cnn2snn.get_akida_version()) 

model_quantized = quantizeml.load_model("heatmap_dsnt_model_untrained_calibrated.h5")
model_quantized.summary()
input_size = (224,224,3)
batch_size = 128
data_path = "data/keypoints.csv"
train_df = pd.read_csv(data_path)
item_list = {col: train_df[col].values for col in train_df.columns}
print(len(item_list['filepath']))

train_dataset = create_dataset(item_list,
                                batch_size=1,
                                input_size=input_size[:2],
                                is_training=True,
                                quantize=True,
                                cache_dir=None)


print("dataset created!")

steps_per_epoch = len(item_list['filepath']) // batch_size



# Define losses and metrics
losses = {"dequantizer_1": heatmap_loss}

metrics_dict = {"dequantizer_1": mpkpe_heatmap}

initial_learning_rate = 1e-3
decay_steps = 5e4  # Adjust based on your dataset size and epochs
decay_rate = 0.96  # Typical value

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)


model_quantized.compile(loss=losses,
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=metrics_dict)

model_quantized.fit(train_dataset, 
                    batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,
                    epochs=30)


model_akida = convert(model_quantized)
model_akida.save("heatmap_dsnt_model")
model_akida.summary()

model_json = model_akida.to_json()
with open('dummy_test_converter.json', 'w') as f:
    json.dump(model_json, f)