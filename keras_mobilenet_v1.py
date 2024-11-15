import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from DataLoading import image_loader
import sys
import json
import logging
from datetime import datetime



logging.basicConfig(stream=sys.stderr, level=logging.INFO)

class PoseEstimationLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_pose, lambda_quat, lambda_norm, name='pose_estimation_loss'):
        super(PoseEstimationLoss, self).__init__(name=name)
        self.lambda_pose = lambda_pose
        self.lambda_quat = lambda_quat
        self.lambda_norm = lambda_norm
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        # Split the predictions and targets into pose and quaternion parameters
        # Extract first 3 values for the pose
        pred_pose = y_pred[:, :3]  # First 3 values for pose
        # Extract last 4 values for the quaternion
        pred_quat = y_pred[:, 3:]  # Last 4 values for quaternion
        
        # Extract first 3 values for the pose in the target
        target_pose = y_true[:, :3]
        # Extract last 4 values for the quaternion in the target
        target_quat = y_true[:, 3:]
        
        # Normalize the predicted quaternion
        pred_quat_norm = pred_quat / tf.norm(pred_quat, axis=1, keepdims=True)
        
        # Pose estimation loss (Mean Squared Error)
        pose_loss = self.mse(target_pose, pred_pose)
        
        # Quaternion regression loss (Mean Squared Error between normalized quaternions)
        quat_loss = self.mse(target_quat, pred_quat_norm)
        
        # Quaternion normalization loss
        quat_norm_loss = tf.reduce_mean((tf.norm(pred_quat, axis=1) - 1) ** 2)
        
        # Total loss with individual weights
        total_loss = (
            self.lambda_pose * pose_loss +
            self.lambda_quat * quat_loss +
            self.lambda_norm * quat_norm_loss
        )
        
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


### DEFINE MODEL
logging.info(tf.config.list_physical_devices())

base_model = MobileNet(include_top=False, alpha=0.7, input_shape=(240, 240, 3), weights=None)
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
    loss=PoseEstimationLoss(1,0,0),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'])

history = model_keras.fit(dataset, epochs=50, verbose=2)
model_keras.save(model_name)
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

json.dump(history, open("history_model.json", 'w'), default=set_default)
logging.info("Training over")