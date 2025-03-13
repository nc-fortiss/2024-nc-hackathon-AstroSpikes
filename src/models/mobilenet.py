import akida_models.imagenet.model_mobilenet as mobilenet
import tensorflow as tf
from cnn2snn import set_akida_version, AkidaVersion
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self, input_size, pretrained=False):
        super(MyModel, self).__init__()

        # Load the base MobileNet model
        with (set_akida_version(AkidaVersion.v1)):  # Keep this if it's necessary for your use case
            if pretrained:
                base_model = mobilenet.mobilenet_imagenet_pretrained(alpha=1.0, quantized=False)
            else:
                base_model = mobilenet.mobilenet_imagenet(input_shape=input_size, alpha=1.0, include_top=False,
                                                          input_scaling=None)

        # Separate layers before and after the layer to be removed
        layers_before = base_model.layers[:-2]  # Layers before the ones to remove

        # Add the layers before the removed ones to the new model
        self.base_model = tf.keras.Sequential()
        for layer in layers_before:
            new_layer = layer
            new_layer.trainable = True
            self.base_model.add(new_layer)

        # Additional layers for the model
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)

        self.pose = tf.keras.layers.Dense(7, activation='linear')

    def call(self, x, training=None, **kwargs):
        # Forward pass through the base model
        x = self.base_model(x)

        # Pass through additional layers
        x = self.global_avg_pooling(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.pose(x)

        # return pos, q_n  # Return the outputs of the two branches
        return x
