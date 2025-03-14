import akida_models.imagenet.model_mobilenet as mobilenet
import tensorflow as tf
from cnn2snn import set_akida_version, AkidaVersion


def MobilenetModel(input_size=(224, 224, 3), pretrained=False):
    with set_akida_version(AkidaVersion.v1):
        if pretrained:
            base_model = mobilenet.mobilenet_imagenet_pretrained(alpha=1.0, quantized=False)
        else:
            base_model = mobilenet.mobilenet_imagenet(input_shape=input_size, alpha=1.0, include_top=False,
                                                      input_scaling=None)

    # Extract feature extractor layers (excluding the last two layers)
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-3].output,
                                       name="feature_extractor")
    feature_extractor.trainable = True  # Allow fine-tuning

    # Input layer
    inputs = tf.keras.Input(shape=input_size, name="image_input")

    # Feature extraction
    x = feature_extractor(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # **Position Branch**
    position_output = tf.keras.layers.Dense(3, activation="linear", name="position_output")(x)

    # **Orientation Branch (Normalized Quaternion)**
    quat_output = tf.keras.layers.Dense(4, activation="linear", name="orientation_raw")(x)
    quat_output = tf.keras.layers.Lambda(lambda q: tf.linalg.l2_normalize(q, axis=1), name="orientation_output")(
        quat_output)

    # Define model with two outputs
    model = tf.keras.Model(inputs=inputs, outputs=[position_output, quat_output], name="MobilenetPoseEstimation")

    return model
