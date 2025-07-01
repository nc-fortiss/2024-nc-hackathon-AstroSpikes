import akida_models.imagenet.model_mobilenet as mobilenet
import tensorflow as tf
from cnn2snn import set_akida_version, AkidaVersion

from akida_models.layer_blocks import Conv2DTranspose, BatchNormalization, ReLU, Conv2D


def MobilenetModelHeatmap(input_shape, num_keypoints, pretrained=True):
    """
    Creates a MobileNet-based model for heatmap regression.

    This model uses a pretrained MobileNet as a feature extractor and then
    applies a series of transposed convolutions (deconvolutions) to upsample
    the feature map and generate a stack of heatmaps.

    Arguments:
        input_shape (tuple): The shape of the input images, e.g., (224, 224, 3).
        num_keypoints (int): The number of keypoints to predict. This will be the
                             number of channels in the output heatmap.
        pretrained (bool): Whether to load weights pretrained on ImageNet for the
                           backbone.
    Returns:
        tf.keras.Model: A Keras model that takes an image and outputs heatmaps.
    """
    with set_akida_version(AkidaVersion.v1):
        if pretrained:
            base_model = mobilenet.mobilenet_imagenet_pretrained(alpha=1.0, quantized=False)
        else:
            base_model = mobilenet.mobilenet_imagenet(input_shape=input_shape, alpha=1.0, include_top=False,
                                                      input_scaling=None)
            # We want to create an upsampling head on top of the backbone
    # Let's make the backbone trainable
    base_model.trainable = True

    # --- Model Definition ---
    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # Feature extraction using the backbone
    x = base_model(inputs, training=True)

    # --- Upsampling Head ---
    # This part takes the small feature map from MobileNetV2 (e.g., 7x7)
    # and upsamples it to a larger heatmap (e.g., 56x56).

    # Upsample block 1
    x = Conv2DTranspose(256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Upsample block 2
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Upsample block 3
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # --- Final Heatmap Layer ---
    # The final layer is a 1x1 convolution with a number of filters equal to
    # the number of keypoints. We output raw logits, as the DSNT function's
    # `_normalise_heatmap` (e.g., via 'softmax') will handle the activation.
    heatmap_logits = Conv2D(
        filters=num_keypoints,
        kernel_size=1,
        padding='same',
        activation=None,  # Output raw scores (logits)
        name='heatmap_logits'
    )(x)

    # Name the output for clarity
    heatmap_output = tf.keras.layers.Activation('linear', name='heatmap_output')(heatmap_logits)

    model = tf.keras.Model(inputs=inputs, outputs=heatmap_output, name="MobilenetV2_Heatmap_Regression")
    return model
