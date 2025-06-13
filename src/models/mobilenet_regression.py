import akida_models.imagenet.model_mobilenet as mobilenet
import tensorflow as tf
from akida_models.layer_blocks import dense_block
from cnn2snn import set_akida_version, AkidaVersion
from keras.layers import Dropout


def MobilenetModel_regression(input_shape, pretrained=False):
    with set_akida_version(AkidaVersion.v1):
        if pretrained:
            base_model = mobilenet.mobilenet_imagenet_pretrained(alpha=1.0, quantized=False)
        else:
            base_model = mobilenet.mobilenet_imagenet(input_shape=input_shape, alpha=1.0, include_top=False,
                                                      input_scaling=None)

    # Extract feature extractor layers
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output,
                                       name="feature_extractor")
    feature_extractor.trainable = True

    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # Feature extraction
    x = feature_extractor(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    px = dense_block(x, units=512, name='fcp1', add_batchnorm=True, relu_activation='ReLU7.5')
    # px = Dropout(0.5, name='dropout_p')(px)
    px = dense_block(px, units=256, name='fcp2', add_batchnorm=True, relu_activation='ReLU7.5')
    position_output = dense_block(px, units=16, name='position', add_batchnorm=False, relu_activation=False)

    model = tf.keras.Model(inputs=inputs, outputs=[position_output], name="MobilenetPoseEstimation")
    return model


if __name__ == '__main__':
    # Initialize model.
    input_shape = (224, 224, 3)
    model = MobilenetModel_regression(input_size=input_shape, pretrained=False)
    model.build(input_shape=(None, *input_shape))  # None is for batch size
    print(model.summary())

    # Print model layers
    print("\nIndividual Layer Summaries:")
    for layer in model.layers:
        # Print individual layer summaries
        print(f"\nLayer Name: {layer.name}")
        print(f"Input Shape: {layer.input_shape}")
        print(f"Output Shape: {layer.output_shape}")
        print(f"Number of Parameters: {layer.count_params()}")
