from cnn2snn import set_akida_version, AkidaVersion
import akida_models.imagenet.model_mobilenet as mobilenet
from tensorflow.keras.models import Model
import tensorflow as tf
import logging
import json
from DataLoading import image_loader


PRETRAINED_MODEL = False
POSITION_DIR = './frames'
EPOCHS = 1000
BATCH_SIZE = 128


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

model_name = "./TR002_model_500epochs_orientation_pose" + str(EPOCHS) + ".keras"
logging.info("Training the model : " + model_name)
with set_akida_version(AkidaVersion.v1):
    if PRETRAINED_MODEL:
        base_model = mobilenet.mobilenet_imagenet_pretrained(alpha=1.0, quantized=False)
        dataset = image_loader.ImageDataLoader(POSITION_DIR, transform=image_loader.ImageDataLoader.center_crop_224x224)()
        
    else:
        base_model = mobilenet.mobilenet_imagenet(input_shape=(224, 224, 3), alpha=1.0, include_top=False, input_scaling=None)
        dataset = image_loader.ImageDataLoader(POSITION_DIR, transform=image_loader.ImageDataLoader.center_crop_224x224)()

    base_model.summary()

    # Separate layers before and after the layer to be removed
    layers_before = base_model.layers[:-2]  # Layers before the ones to remove

    model_keras = tf.keras.models.Sequential()

    # Add layers before the one to be removed
    for layer in layers_before:
        new_layer = layer
        new_layer.trainable = True
        # print("layer ", layer, layer.name, len(layer.trainable_weights))
        # print("new_layer ", new_layer, new_layer.name, len(new_layer.trainable_weights))
        model_keras.add(new_layer)

    model_keras.add(tf.keras.layers.Flatten())
    model_keras.add(tf.keras.layers.Dense(7, activation='linear'))
    
    ### DEFINE MODEL
    model_keras.compile(loss=PoseEstimationLoss(0.6 ,0.3, 0.1),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        metrics=['accuracy'])
    
logging.info(model_keras.summary())

# model_keras.summary()
# print("Model compiled successfully!")
### TRAIN MODEL
history = model_keras.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)    

model_keras.save(model_name)
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

json.dump(history, open("history_model.json", 'w'), default=set_default)
logging.info("Training over")