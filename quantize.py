from quantizeml.models import quantize, QuantizationParams
import tensorflow as tf
from tensorflow.keras.models import Model
from losses import PoseEstimationLoss

FORCE_FULL_QUANTIZATION = True

tf.keras.utils.get_custom_objects().update({"PoseEstimationLoss": PoseEstimationLoss})
loaded_model = tf.keras.models.load_model("latest_model_3231.keras")
print("Model loaded successfully!")
loaded_model.summary()

if FORCE_FULL_QUANTIZATION:
    # layer = model.layers[74]
    # print(layer.get_config())

    # Separate layers before and after the layer to be removed
    layers_before = loaded_model.layers[:73]  # Layers before the one to remove
    layers_after = loaded_model.layers[74:]   # Layers after the one to remove

    # Start building the new model with the same input
    inputs = loaded_model.input
    x = inputs

    # Add layers before the one to be removed
    for layer in layers_before:
        print(layer.name)
        x = layer(x)

    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense((16*16, 358), activation='linear')(x)
    # x = tf.keras.layers.Reshape((16, 16, 358))(x)

    # # After conv_pw_11_relu (15x15x358)
    # x = tf.keras.layers.Conv2D(
    #     filters=358,
    #     kernel_size=(1, 1),  # 1x1 convolution
    #     strides=(1, 1),
    #     padding='same',
    #     use_bias=False
    # )(x)

    # x = tf.keras.layers.Conv2D(
    #     filters=358,
    #     kernel_size=(2, 2),
    #     strides=(1, 1),
    #     padding='same',
    #     use_bias=False
    # )(x)  # Output: 16x16x358

    # Skip the 74th layer and continue with subsequent layers
    for layer in layers_after[:]:
        print(layer.name)
        x = layer(x)

    # Create the new model
    model = Model(inputs=inputs, outputs=x)

    # Compile the new model with the same settings as the original if needed
    model.compile(optimizer=loaded_model.optimizer, loss=loaded_model.loss)
else:
    model = loaded_model


# Verify the new model structure
model.summary()

qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
# TODO: call quantize with the real training data
# https://doc.brainchipinc.com/examples/general/plot_0_global_workflow.html#sphx-glr-examples-general-plot-0-global-workflow-py
# model_quantized = quantize(model_keras, qparams=qparams,
                        #    samples=x_train, num_samples=1024, batch_size=100, epochs=2)
# model_quantized = quantize(model, qparams=qparams)
model_quantized = quantize(model, qparams=qparams)
model_quantized.summary()