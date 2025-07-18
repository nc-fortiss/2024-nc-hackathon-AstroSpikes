from akida_models.imagenet.imagenet_utils import obtain_input_shape
from akida_models.layer_blocks import conv_block, separable_conv_block
from akida_models.utils import get_params_by_version
from keras import Model, regularizers
from keras.layers import Input, Rescaling


def mobilenet_heatmap_compact(input_size=None,
                              alpha=1.0,
                              num_keypoints=8,
                              dropout=1e-3,
                              include_top=False,
                              pooling=None,
                              classes=1000,
                              use_stride2=True,
                              input_scaling=None):
    """Instantiates the MobileNet architecture.

    Note: input preprocessing is included as part of the model (as a Rescaling layer). This model
    expects inputs to be float tensors of pixels with values in the [0, 255] range.

    Args:
        input_size (tuple, optional): shape tuple. Defaults to None.
        alpha (float, optional): controls the width of the model.
            Defaults to 1.0.

            * If `alpha` < 1.0, proportionally decreases the number of filters
              in each layer.
            * If `alpha` > 1.0, proportionally increases the number of filters
              in each layer.
            * If `alpha` = 1, default number of filters from the paper are used
              at each layer.
        dropout (float, optional): dropout rate. Defaults to 1e-3.
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the model. Defaults to True.
        pooling (str, optional): optional pooling mode for feature extraction
            when `include_top` is `False`.
            Defaults to None.

            * `None` means that the output of the model will be the 4D tensor
              output of the last convolutional block.
            * `avg` means that global average pooling will be applied to the
              output of the last convolutional block, and thus the output of the
              model will be a 2D tensor.
        classes (int, optional): optional number of classes to classify images
            into, only to be specified if `include_top` is `True`. Defaults to 1000.
        use_stride2 (bool, optional): replace max pooling operations by stride 2
            convolutions in layers separable 2, 4, 6 and 12. Defaults to True.
        input_scaling (tuple, optional): scale factor and offset to apply to
            inputs. Defaults to (128, -1). Note that following Akida convention,
            the scale factor is an integer used as a divisor.

    Returns:
        keras.Model: a Keras model for MobileNet/ImageNet.

    Raises:
        ValueError: in case of invalid input shape.
    """
    # Model version management
    # Model version management
    fused, post_relu_gap, relu_activation = get_params_by_version()
    # fused, post_relu_gap, relu_activation = get_params_by_version(relu_v2='ReLU7.5')

    # Define weight regularization, will apply to the first convolutional layer
    # and to all pointwise weights of separable convolutional layers.
    weight_regularizer = regularizers.l2(4e-5)

    # Define stride 2 or max pooling
    if use_stride2:
        sep_conv_pooling = None
        strides = 2
    else:
        sep_conv_pooling = 'max'
        strides = 1

    # Determine proper input shape and default size.
    if input_size is None:
        default_size = 224
    else:
        rows = input_size[0]
        cols = input_size[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_size = obtain_input_shape(input_size,
                                    default_size=default_size,
                                    min_size=32,
                                    include_top=include_top)

    rows = input_size[0]
    cols = input_size[1]

    img_input = Input(shape=input_size, name="input")

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
                   strides=1,  # 2,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   kernel_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(64 * alpha),
                             name='separable_1',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(128 * alpha),
                             name='separable_2',
                             kernel_size=(3, 3),
                             padding='same',
                             pooling=sep_conv_pooling,
                             strides=strides,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(128 * alpha),
                             name='separable_3',
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(256 * alpha),
                             name='separable_4',
                             kernel_size=(3, 3),
                             padding='same',
                             pooling=sep_conv_pooling,
                             strides=1,  # strides,
                             use_bias=False,
                             add_batchnorm=True,
                             relu_activation=relu_activation,
                             fused=fused,
                             pointwise_regularizer=weight_regularizer)

    x = separable_conv_block(x,
                             filters=int(512 * alpha),
                             name='separable_5',
                             kernel_size=(3, 3),
                             padding='same',
                             pooling=sep_conv_pooling,
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
                             strides=strides,
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

    # Last separable layer with global pooling
    x = conv_block(x,
                   filters=int(256 * alpha),
                   name='conv256',
                   kernel_size=(3, 3),
                   padding='same',
                   pooling='avg',
                   use_bias=False,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   post_relu_gap=post_relu_gap)
    x = conv_block(x,
                   filters=int(64 * alpha),
                   name='conv64',
                   kernel_size=(3, 3),
                   padding='same',
                   pooling='avg',
                   use_bias=False,
                   add_batchnorm=True,
                   relu_activation=relu_activation,
                   post_relu_gap=post_relu_gap)
    x = conv_block(x,
                   filters=int(num_keypoints * alpha),
                   name='heatmap_output',
                   kernel_size=(1, 1),
                   padding='same',
                   pooling='avg',
                   use_bias=False,
                   add_batchnorm=False,
                   relu_activation=False,
                   post_relu_gap=post_relu_gap)

    # Create model.
    return Model(img_input, x, name='mobilenet_%0.2f_%s_%s' % (alpha, rows, classes))
