import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import sys

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


### DEFINE MODEL

base_model = MobileNet(include_top=False, alpha=0.7, input_shape=(240, 240, 3), weights=None)
# base_model.trainable = False  # Freeze all layers in the base model

x = base_model.output
# x = tf.keras.layers.Reshape((7*7, 1024))(x)
# x = tf.keras.layers.Dense(1, activation='linear')(x)
x = tf.keras.layers.Flatten()(x)
output_l = tf.keras.layers.Dense(7, activation='linear')(x)
model_keras = Model(inputs=base_model.input, outputs=output_l)

print(model_keras.summary())

### TRAIN MODEL

#TODO: load data
position_dir = './event_frames'
# position_file = 'position.csv'
# # Load position data from CSV
y_train = tf.convert_to_tensor(list(range(7)), dtype=tf.float32)
y_train = tf.expand_dims(y_train, axis=1)
y_train = tf.tile(y_train, [1, 200])
y_train = tf.transpose(y_train, perm=[1, 0])
print(y_train.shape)
# with open(os.path.abspath(os.path.join(position_dir, position_file))) as file:
#     reader = csv.reader(file)
#     for row in reader:
#         y_test.append(row)
        
# Load images
x_train = []
for _ in range(20):
    for i in range(10):
        img = plt.imread(f'{position_dir}/frame_{i:02}.png',)
        x_train.append(img)
    
    
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)


model_keras.compile(
    loss=PoseEstimationLoss(0.5,0.3,0.2),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy'])

_ = model_keras.fit(x_train, y_train, epochs=10, validation_split=0.1)

model_keras.save("./latest_model.keras")