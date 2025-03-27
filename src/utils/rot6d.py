'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
The implementation of 6D rotatiton representation,
based on

    https://arxiv.org/abs/1812.07035

"On the continuity of rotation representations in neural networks"
Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.
Conference on Neural Information Processing Systems (NeurIPS) 2019.
'''
import tensorflow as tf


def tf_rotation6d_to_matrix(r6d):
    """Compute rotation matrix from 6D rotation representation.
    Implementation based on https://arxiv.org/abs/1812.07035
    [Inputs]
        6D rotation representation (last dimension is 6)
    [Returns]
        flattened rotation matrix (last dimension is 9)
    """
    tensor_shape = r6d.shape.as_list()  # Use r6d.shape instead of r6d.get_shape()

    if not tensor_shape[-1] == 6:
        raise AttributeError(
            "The last dimension of the inputs in tf_rotation6d_to_matrix should be 6, "
            "but found tensor with shape {}".format(tensor_shape[-1])
        )

    r6d = tf.reshape(r6d, [-1, 6])
    x_raw = r6d[:, 0:3]
    y_raw = r6d[:, 3:6]

    x = tf.linalg.normalize(x_raw, axis=-1)[0]  # Use tf.linalg.normalize and get the output tensor
    z = tf.linalg.normalize(tf.linalg.cross(x, y_raw), axis=-1)[0]  # Normalize within cross product
    y = tf.linalg.normalize(tf.linalg.cross(z, x), axis=-1)[0]

    x = tf.reshape(x, [-1, 3, 1])
    y = tf.reshape(y, [-1, 3, 1])
    z = tf.reshape(z, [-1, 3, 1])
    matrix = tf.concat([x, y, z], axis=-1)

    if len(tensor_shape) == 1:
        matrix = tf.reshape(matrix, [9])
    else:
        output_shape = tensor_shape[:-1] + [9]
        matrix = tf.reshape(matrix, output_shape)

    return matrix


def tf_matrix_to_rotation6d(mat):
    """Get 6D rotation representation for rotation matrix.
    Implementation based on https://arxiv.org/abs/1812.07035
    [Inputs]
        flattened rotation matrix (last dimension is 9)
    [Returns]
        6D rotation representation (last dimension is 6)
    """
    tensor_shape = mat.shape.as_list()

    if not ((tensor_shape[-1] == 3 and len(tensor_shape) > 1 and tensor_shape[-2] == 3) or (tensor_shape[-1] == 9)):
        raise AttributeError(
            "The inputs in tf_matrix_to_rotation6d should be [...,3,3] or [...,9], "
            "but found tensor with shape {}".format(tensor_shape)
        )

    if tensor_shape[-1] == 9:
        mat = tf.reshape(mat, [-1, 3, 3])  # Reshape only if input is flattened

    r6d = tf.concat([mat[..., 0], mat[..., 1]], axis=-1)

    if len(tensor_shape) == 1:
        r6d = tf.reshape(r6d, [6])
    else:
        output_shape = tensor_shape[:-1] + [6]
        r6d = tf.reshape(r6d, output_shape)  # reshape after concatenate

    return r6d
