from keras import Model
from keras.layers import Input
from akida_models.layer_blocks import conv_block, separable_conv_block
from akida_models.imagenet.imagenet_utils import obtain_input_shape
from akida_models.utils import get_params_by_version
from keras.layers import Rescaling


def mobilenet_heatmap_akida(input_size=(224, 224, 3),
                            alpha=1.0,
                            num_keypoints=8,
                            input_scaling=(128, -1)):
    """MobileNet variant for Akida that outputs heatmaps."""

    fused, post_relu_gap, relu_activation = get_params_by_version()

    input_size = obtain_input_shape(input_size,
                                    default_size=224,
                                    min_size=32,
                                    include_top=False)

    img_input = Input(shape=input_size, name="input")

    if input_scaling:
        scale, offset = input_scaling
        x = Rescaling(1. / scale, offset, name="rescaling")(img_input)
    else:
        x = img_input

    # Initial conv (224x224x3 -> 224x224x32)
    x = conv_block(x, filters=int(32 * alpha), kernel_size=(3, 3), strides=1,
                   name='conv_0', use_bias=False, relu_activation=relu_activation)

    # Depthwise stack
    # (224x224x32 -> 112x112x64)
    x = separable_conv_block(x, filters=int(64 * alpha), strides=2,
                             name='sep_1', kernel_size=(3, 3), fused=False, relu_activation=relu_activation)

    # (112x112x64 -> 112x112x128)
    x = separable_conv_block(x, filters=int(128 * alpha), strides=1,
                             name='sep_2', kernel_size=(3, 3), fused=False, relu_activation=relu_activation)

    # (112x112x128 -> 56x56x128)
    x = separable_conv_block(x, filters=int(128 * alpha), strides=2,
                             name='sep_3', kernel_size=(3, 3), fused=False, relu_activation=relu_activation)

    # (56x56x128 -> 56x56x256)
    x = separable_conv_block(x, filters=int(256 * alpha), strides=1,
                             name='sep_4', kernel_size=(3, 3), fused=False, relu_activation=relu_activation)

    # Add heatmap head at 56x56 resolution
    heatmap = conv_block(x, filters=num_keypoints, kernel_size=(1, 1), strides=1,
                               name='heatmap_output', relu_activation=False, use_bias=False)

    return Model(inputs=img_input, outputs=heatmap, name='mobilenet_heatmap')