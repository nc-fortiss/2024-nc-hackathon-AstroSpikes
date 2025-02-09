from cnn2snn import set_akida_version, AkidaVersion
import akida_models.imagenet.model_mobilenet as mobilenet
from tensorflow.keras.models import Model
import tensorflow as tf
import logging
import json
from DataLoading import image_loader
from datetime import datetime
import wandb
from omegaconf import OmegaConf
import sys


log_wandb = True

wandb.login(key='d8eb14aa69c0f4a4cc666324156979070f9ccb7b')
try:
    config = OmegaConf.load("conf/global_conf.yaml")
    print(config)
except Exception as e:
    print("Error loading YAML:", e)



PRETRAINED_MODEL = False
POSITION_DIR_TRAIN = config.paths.output_dir
POSITION_DIR_TEST = config.paths.output_dir
EPOCHS = config.epochs
BATCH_SIZE = config.batch_size

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


class WandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epochs, logs=None):
        dict_keys = logs.keys()
        keys = list(dict_keys)
        logging.info("...Training: start of batch {}; got log keys: {}".format("0", keys))
        wandb.log({"epoch": "a2","acc": 1, "loss": 1})

model_name = "./model_pretrained" + datetime.now().strftime("%Y%m%d_%H%M_") + ".keras"

logging.info("Training the model : " + model_name)

with set_akida_version(AkidaVersion.v1):
    if PRETRAINED_MODEL:
        base_model = mobilenet.mobilenet_imagenet_pretrained(alpha=1.0, quantized=False)
        
    else:
        base_model = mobilenet.mobilenet_imagenet(input_shape=(224, 224, 3), alpha=1.0, include_top=False, input_scaling=None)

    # Load the dataset
    data_loader = image_loader.ImageDataLoader(test=True)
    train_dataset, test_dataset = data_loader()

    # train_dataset = image_loader.ImageDataLoader(test=False) #, transform=image_loader.ImageDataLoader.center_crop_224x224)()
    # test_dataset = image_loader.ImageDataLoader(test=True) #, transform=image_loader.ImageDataLoader.center_crop_224x224)()
    # Separate layers before and after the layer to be removed
    layers_before = base_model.layers[:-2]  # Layers before the ones to remove
    
    logging.info("Loading dataset")
    sys.stdout.flush()
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
    def scheduler(epoch, lr):
        if epoch < 400:
            return 1e-4
        elif epoch < 800 :
            return 1e-5
        else :
            return 1e-5
        
    layers_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/" + model_name,
    monitor='val_loss',
    verbose=config.verbose,
    save_best_only=True,
    mode='min',
    save_freq='epoch',
    initial_value_threshold=None
)
    model_keras.load_weights("/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/model_20241203_1634_.keras")
    model_keras.compile(loss=PoseEstimationLoss(0.6,0.3, 0.1),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                        metrics=[config.metrics[0]])
    
logging.info(model_keras.summary())
logging.info("Training model " + model_name)

if log_wandb:
    wandb.init(# set the wandb project where this run will be logged
        project="mobilenet-astrospikes",
        #set the transformation
        notes = str(config.transformation.method),
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(config))

callbacks = [layers_callback, checkpoint_callback, WandbCallback()] if  log_wandb else [layers_callback, checkpoint_callback] 

### TRAIN MODEL
history = model_keras.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE,  \
                          callbacks=callbacks, verbose=2, shuffle=True, validation_data=test_dataset)    

if log_wandb :
    wandb.finish()

model_keras.save(model_name)
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

json.dump(history, open("history_model.json", 'w'), default=set_default)
logging.info("Training over")