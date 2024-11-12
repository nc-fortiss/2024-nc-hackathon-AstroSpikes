import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model

import sys

### DEFINE MODEL

base_model = MobileNet(include_top=False, input_shape=(240, 240, 2), weights=None)
# base_model.trainable = False  # Freeze all layers in the base model

x = base_model.output
# x = tf.keras.layers.Reshape((7*7, 1024))(x)
# x = tf.keras.layers.Dense(1, activation='linear')(x)
x = tf.keras.layers.Flatten()(x)
output_l = tf.keras.layers.Dense(7, activation='linear')(x)
model_keras = Model(inputs=base_model.input, outputs=output_l)

print(model_keras.summary())

### TRAIN MODEL

#TODO: load data


model_keras.compile(
    loss= None, #TODO: add our loss
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'])

_ = model_keras.fit(x_train, y_train, epochs=10, validation_split=0.1)


### QUANTIZE

from quantizeml.models import quantize, QuantizationParams

qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
# TODO: call quantize with the real training data
# https://doc.brainchipinc.com/examples/general/plot_0_global_workflow.html#sphx-glr-examples-general-plot-0-global-workflow-py
# model_quantized = quantize(model_keras, qparams=qparams,
                        #    samples=x_train, num_samples=1024, batch_size=100, epochs=2)
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()

# TODO: evaluate model
def compile_evaluate(model):
    """ Compiles and evaluates the model, then return accuracy score. """
    model.compile(metrics=['accuracy'])
    return model.evaluate(x_test, y_test, verbose=0)[1]


print('Test accuracy after 8-bit quantization:', compile_evaluate(model_quantized))

model_quantized.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy'])

# score = model_quantized.evaluate(x_test, y_test, verbose=0)[1]
# print('Test accuracy after fine tuning:', score)

# model_quantized.fit(x_train, y_train, epochs=5, validation_split=0.1)



### CONVERT TO AIKIDA NETWORK

from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()


# # check performance
# accuracy = model_akida.evaluate(x_test, y_test)
# print('Test accuracy after conversion:', accuracy)

# # For non-regression purposes
# assert accuracy > 0.96