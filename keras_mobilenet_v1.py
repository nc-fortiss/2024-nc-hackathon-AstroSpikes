import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from DataLoading import image_loader
import sys
import json
import logging
from datetime import datetime

from omegaconf import OmegaConf

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

import tensorflow as tf

class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, initial_S=1.0, name='pose_estimation_loss'):
        """
        Custom Pose Estimation Loss.

        Parameters:
        - lambda_pose: Weight for pose loss (MSE of positions).
        - lambda_quat: Weight for quaternion loss.
        - initial_S: Initial value for the scaling factor S.
        - name: Name of the loss function.
        """
        super(PoseEstimationLoss, self).__init__(name=name)
        # S is a trainable variable initialized with initial_S
        self.S = tf.Variable(initial_S, trainable=True, dtype=tf.float32, name="scaling_factor_S")
        self.mse = tf.keras.losses.MeanSquaredError()

    def quanterror(self, y_true, y_pred):
        """
        Compute the orientation error between predicted and ground truth quaternions.

        Parameters:
        - y_true: Tensor of ground truth values (including quaternion at the last 4 values).
        - y_pred: Tensor of predicted values (including quaternion at the last 4 values).

        Returns:
        - Orientation error in radians (as a tensor).
        """
        pred_quat = y_pred[:, 3:]
        target_quat = y_true[:, 3:]

        # Normalize quaternions
       # pred_quat = tf.nn.l2_normalize(pred_quat, axis=-1)
       # target_quat = tf.nn.l2_normalize(target_quat, axis=-1)

        # Dot product and clamp to avoid numerical instability
        dot_product = tf.reduce_sum(pred_quat * target_quat, axis=-1)
        dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)

        # Compute orientation error
        orientation_error = 2 * tf.acos(tf.abs(dot_product))
        return tf.reduce_mean(orientation_error)

    def call(self, y_true, y_pred):
        """
        Compute the total loss as a weighted sum of pose loss and quaternion loss.

        Parameters:
        - y_true: Ground truth tensor [batch_size, 7].
        - y_pred: Predicted tensor [batch_size, 7].

        Returns:
        - Total loss (scalar tensor).
        """
        pred_pose = y_pred[:, :3]
        target_pose = y_true[:, :3]

        # Pose loss
        pose_loss = self.mse(target_pose, pred_pose)

        # Quaternion orientation loss
        quat_loss = self.quanterror(y_true, y_pred)

        # Total loss with trainable S
        total_loss = pose_loss + self.S * quat_loss
        return total_loss


model_name = "./model_" + datetime.now().strftime("%Y%m%d_%H%M_") + ".keras"
class CustomModel(tf.keras.Model):
    def __init__(self, save_period=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_period = save_period
        self.train_count = 0

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.train_count +=1
        if self.train_count%self.save_period==0:
            logging.info("Saving model " + model_name)
            self.save(model_name)
        return {m.name: m.result() for m in self.metrics}


conf_file = "conf/global_conf.yaml"

### DEFINE MODEL
logging.info(tf.config.list_physical_devices())

base_model = MobileNet(include_top=False, alpha=conf_file.alpha, input_shape=tuple(conf_file.input_shape), weights=None)
# base_model.trainable = False  # Freeze all layers in the base model

x = base_model.output
# x = tf.keras.layers.Reshape((7*7, 1024))(x)
# x = tf.keras.layers.Dense(1, activation='linear')(x)
x = tf.keras.layers.Flatten()(x)
output_l = tf.keras.layers.Dense(7, activation='linear')(x)
model_keras = CustomModel(inputs=base_model.input, outputs=output_l)

logging.info(model_keras.summary())

### TRAIN MODEL

#TODO: load data
position_dir = './generating_dataset'
# position_file = 'position.csv'
# # Load position data from CSV
dataset = image_loader.ImageDataLoader(position_dir)()
# with open(os.path.abspath(os.path.join(position_dir, position_file))) as file:
#     reader = csv.reader(file)
#     for row in reader:
#         y_test.append(row)

model_keras.compile(
    loss=PoseEstimationLoss(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=conf_file.learning_rate),
    metrics=conf_file.metrics)

history = model_keras.fit(dataset, epochs=conf_file.epochs, verbose=conf_file.verbose)
model_keras.save(model_name)
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

json.dump(history, open("history_model.json", 'w'), default=set_default)
logging.info("Training over")