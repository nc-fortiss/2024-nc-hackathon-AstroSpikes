import tensorflow as tf
import logging
import os
import logging
import akida as ak
import cnn2snn
import sys

from src.models.mobilenet_heatmap import mobilenet_heatmap_7pass, mobilenet_heatmap_1pass
import keras

os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = "v1"
print("\nCONFIGURATION")
print("TensorFlow version: ", tf.__version__)
print("    MetaTF version: ", ak.__version__)
print('     Akida version: ', cnn2snn.get_akida_version())

tf.config.run_functions_eagerly(False)

virtual_device = ak.AKD1500()
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# Loading from a model class:
model = mobilenet_heatmap_1pass(input_size=(224, 224, 3), num_keypoints=8)
model.build(input_shape=(None, *(224, 224, 3)))  # None is for batch size

# Loading from a .keras trained model:
# model_name = "networks/mobilenet_heatmap_layer_test.keras"
# model = keras.models.load_model(model_name)

model.summary()
cnn2snn.check_model_compatibility(model, device=virtual_device)

# If an error is thrown at a specific layer, it  means that the previous layer is too large to fit on the chip.
# If an error is thrown at the end with: Incompatibility found during mapping: Model cannot be mapped in one hardware sequence. it means that the network cannot fit in one sequence but that the Hardware Partial Reconfiguration is possible with AKD1500
# If there is no error, it means that the network can fit in one sequence and be very fast and power efficient.
