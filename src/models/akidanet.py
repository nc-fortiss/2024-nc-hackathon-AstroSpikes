from akida_models.imagenet.imagenet_utils import obtain_input_shape
from akida_models.layer_blocks import conv_block, separable_conv_block
from akida_models.utils import get_params_by_version
from keras import Model, regularizers
from keras.layers import Input, Rescaling


def akidanet_imagenet(input_shape=None,
                      alpha=1.0,
                      include_top=True,
                      pooling=None,
                      num_keypoints=8,
                      classes=1000,
                      input_scaling=(128, -1)):
    """Instantiates the AkidaNet architecture.

    Note: input preprocessing is included as part of the model (as a Rescaling layer). This model
    expects inputs to be float tensors of pixels with values in the [0, 255] range.

    Args:
        input_shape (tuple, optional): shape tuple. Defaults to None.
        alpha (float, optional): controls the width of the model.
            Defaults to 1.0.

            * If `alpha` < 1.0, proportionally decreases the number of filters
              in each layer.
            * If `alpha` > 1.0, proportionally increases the number of filters
              in each layer.
            * If `alpha` = 1, default number of filters from the paper are used
              at each layer.

    Returns:
        keras.Model: a Keras model for AkidaNet/ImageNet.

    Raises:
        ValueError: in case of invalid input shape.
    """
    # Define weight regularization, will apply to the convolutional layers and
    # to all pointwise weights of separable convolutional layers.
    weight_regularizer = regularizers.l2(4e-5)

    # Model version management
    fused, post_relu_gap, relu_activation = get_params_by_version(relu_v2='ReLU7.5')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        rows = input_shape[0]
        cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = obtain_input_shape(input_shape,
                                     default_size=default_size,
                                     min_size=32,
                                     include_top=include_top)

    img_input = Input(shape=input_shape, name="input")

    if input_scaling is None:
        x = img_input
    else:
        scale, offset = input_scaling
        x = Rescaling(1. / scale, offset, name="rescaling")(img_input)

    x = conv_block(x,
                   filters=int(32 * alpha),
                   name='conv_0',
                   kernel_size=(3, 3),
                   padding='same',
                   use_bias=False,
                   strides=2,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = conv_block(x,
                   filters=int(64 * alpha),
                   name='conv_1',
                   kernel_size=(3, 3),
                   padding='same',
                   use_bias=False,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = conv_block(x,
                   filters=int(128 * alpha),
                   name='conv_2',
                   kernel_size=(3, 3),
                   padding='same',
                   strides=2,
                   use_bias=False,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = conv_block(x,
                   filters=int(128 * alpha),
                   name='conv_3',
                   kernel_size=(3, 3),
                   padding='same',
                   use_bias=False,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(256 * alpha),
                             name='separable_4',
                             kernel_size=(3, 3),
                             padding='same',
                             strides=2,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(256 * alpha),
                             name='separable_5',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_6',
                             kernel_size=(3, 3),
                             padding='same',
                             strides=2,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_7',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_8',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_9',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_10',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_11',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(1024 * alpha),
                             name='separable_12',
                             kernel_size=(3, 3),
                             padding='same',
                             strides=2,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    # Last separable layer with global pooling
    # layer_pooling = 'global_avg' if include_top or pooling == 'avg' else None
    x = separable_conv_block(x,
                             filters=int(1024 * alpha),
                             name='separable_13',
                             kernel_size=(3, 3),
                             padding='same',
                             pooling=None,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=False,
                             fused=fused,
                             post_relu_gap=post_relu_gap,
                             pointwise_regularizer=weight_regularizer)
    
    x = separable_conv_block(x,
                             filters=int(num_keypoints * alpha),
                             name='separable_14',
                             kernel_size=(3, 3),
                             padding='same',
                             pooling=None,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=False,
                             fused=fused,
                             post_relu_gap=post_relu_gap,
                             pointwise_regularizer=weight_regularizer)

    return Model(img_input, x, name='akidanet_%0.2f_%s_%s' % (alpha, input_shape[0], classes))


if __name__ == '__main__':
    # Initialize model.
    input_shape = (224, 224, 3)
    model = akidanet_imagenet(input_shape=input_shape)
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
