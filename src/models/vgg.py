from akida_models.layer_blocks import conv_block, dense_block
from akida_models.utils import get_params_by_version
from keras import Model
from keras.layers import Dropout, Input, Rescaling

import akida_models.imagenet.model_akidanet


def vgg_heatmap(input_shape=(32, 32, 3), input_scaling=(127, -1)):
    """Instantiates a VGG-like model for the regression example on age
    estimation using UTKFace dataset.

    Note: input preprocessing is included as part of the model (as a Rescaling layer). This model
    expects inputs to be float tensors of pixels with values in the [0, 255] range.

    Args:
        input_shape (tuple, optional): input shape tuple of the model. Defaults to (32, 32, 3).
        input_scaling (tuple, optional): scale factor and offset to apply to
            inputs. Defaults to (127, -1). Note that following Akida convention,
            the scale factor is an integer used as a divisor.

    Returns:
        keras.Model: a Keras model for VGG/UTKFace
    """
    img_input = Input(shape=input_shape, name="input")

    if input_scaling is None:
        x = img_input
    else:
        scale, offset = input_scaling
        x = Rescaling(1. / scale, offset, name="rescaling")(img_input)

    # Model version management
    _, post_relu_gap, relu_activation = get_params_by_version()

    x = conv_block(x,
                   filters=32,
                   kernel_size=(3, 3),
                   name='conv_0',
                   use_bias=False,
                   relu_activation=relu_activation,
                   add_batchnorm=True)

    x = conv_block(x,
                   filters=32,
                   kernel_size=(3, 3),
                   name='conv_1',
                   padding='same',
                   pooling='max',
                   pool_size=2,
                   use_bias=False,
                   relu_activation=relu_activation,
                   add_batchnorm=True)

    x = Dropout(0.3, name="dropout_3")(x)

    x = conv_block(x,
                   filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   name='conv_2',
                   use_bias=False,
                   relu_activation=relu_activation,
                   add_batchnorm=True)

    x = conv_block(x,
                   filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   name='conv_3',
                   pooling='max',
                   pool_size=2,
                   use_bias=False,
                   relu_activation=relu_activation,
                   add_batchnorm=True)

    x = Dropout(0.3, name="dropout_4")(x)

    x = conv_block(x,
                   filters=84,
                   kernel_size=(3, 3),
                   padding='same',
                   name='conv_4',
                   use_bias=False,
                   relu_activation=relu_activation,
                   pooling=None,
                   post_relu_gap=post_relu_gap,
                   add_batchnorm=True)

    # x = Dropout(0.3, name="dropout_5")(x)
    #
    # x = dense_block(x,
    #                 units=64,
    #                 name='dense_1',
    #                 use_bias=False,
    #                 relu_activation=relu_activation,
    #                 add_batchnorm=True)
    #
    # x = dense_block(x, units=1, name='dense_2', relu_activation=False)

    return Model(img_input, x, name='vgg')


if __name__ == '__main__':
    # Initialize model.
    input_shape = (224, 224, 3)
    model = vgg_heatmap(input_shape=input_shape)
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