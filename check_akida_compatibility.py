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
from quantizeml.models import quantize, QuantizationParams
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

qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
model_name = "model_38_0.0107.keras"
input_size = (224,224,3)
output_directory ='/home/lecomte/Documents/spikingbody/output_front/'
model = keras.models.load_model(model_name)
#model = mobilenet_imagenet(input_shape=input_size)
model.summary()
cnn2snn.check_model_compatibility(model)

print("+++++++++++++++++Model Compatible++++++++++++++++++++++++")

model.build(input_shape=(None, *list(input_size)))  # None is for batch size
#model.load_weights(model_name)

""" train_dataset = create_dataset(output_directory,
                                batch_size=1,
                                input_size=input_size[:2],
                                is_training=True,
                                cache_dir=None)

print(len(train_dataset))

dummy_input, dummy_target = next(iter(train_dataset))
print(type(dummy_input)) """
#cnn2snn.check_model_compatibility(model)
model_quantized = quantize(model, qparams=qparams)
print("++++++++++++++++++MODEL QUANTIZED+++++++++++++++++++++++++++++")
#resolution = (dummy_input.shape[1], dummy_input.shape[2])
#dummy_input = dummy_input.to(device, dtype=torch.float)

#dummy_output = model.predict(dummy_input) #network initialization
#logging.info(dummy_output)

#model_quantized.compile(
#    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    optimizer=Adam(learning_rate=1e-4),
#    metrics=['accuracy'])

#model_quantized.fit(x_train, y_train, epochs=5, validation_split=0.1)


model_akida = convert(model_quantized)
model_akida.save("heatmap_2.12model")
model_akida.summary()

model_json = model_akida.to_json()
with open('dummy_test_converter.json', 'w') as f:
    json.dump(model_json, f)