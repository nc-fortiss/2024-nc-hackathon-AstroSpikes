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
    def __init__(self, beta, name='pose_estimation_loss'):
        super(PoseEstimationLoss, self).__init__(name=name)
        self.beta = beta  # Beta is now a fixed scaling factor for each run
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        pred_pose = y_pred[:, :3]  # First 3 values for position
        pred_quat = y_pred[:, 3:]  # Last 4 values for quaternion
        
        target_pose = y_true[:, :3]
        target_quat = y_true[:, 3:]

        epsilon = 1e-7
        # Normalize the predicted quaternion
        pred_quat_norm = pred_quat / (tf.norm(pred_quat, axis=1, keepdims=True) + epsilon)

        # Compute position loss (MSE)
        pose_loss_tensor = self.mse(target_pose, pred_pose)

        # Compute orientation loss (Quaternion MSE)
        quat_loss_tensor = self.mse(target_quat, pred_quat_norm)

        # Total loss as a linear combination of the two losses with scaling factor beta
        total_loss = pose_loss_tensor + self.beta * quat_loss_tensor

        # wandb.log({
        #     "pose_loss": float(pose_loss_tensor.numpy()),
        #     "quat_loss": float(quat_loss_tensor.numpy()),
        #     "beta_ratio": float((pose_loss_tensor / (quat_loss_tensor + epsilon)).numpy())
        # })

        return total_loss

class CheckWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.weight_differences_over_time = []

    def on_test_begin(self, logs=None):
        self.weights_before = [layer.get_weights() for layer in self.model.layers]
        print("Weights captured on_test_begin")

    def on_test_end(self, logs=None):
        weights_after = [layer.get_weights() for layer in self.model.layers]
        differences = []

        # Compare weights before and after
        for i, (before, after) in enumerate(zip(self.weights_before, weights_after)):
            if len(before) > 0:  # Only check layers with weights
                difference = np.mean([np.mean(np.abs(b - a)) for b, a in zip(before, after)])
                differences.append(difference)
                print(f"Weight difference for layer {i}: {difference}")

        # Calculate the mean difference across all layers for this phase
        mean_difference = np.mean(differences)
        self.weight_differences_over_time.append(mean_difference)
        
        # Log the weight differences as a line plot to WandB
        wandb.log({"weight_differences_line": self.weight_differences_over_time})
        print("Weights checked on_test_end")

class PrintLabelsCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🚀 Epoch {epoch + 1} - First 10 training labels:")

        # Fetch first batch of training data
        for batch in self.model.train_dataset.take(1):
            labels = batch[1].numpy()[:10]  # First 5 labels
            print(labels)
            break  # Exit after printing the first batch

    def on_test_begin(self, logs=None):
        print(f"\n📊 Validation - First 10 testing labels:")

        # Fetch first batch of validation data
        for batch in self.model.validation_data.take(1):
            labels = batch[1].numpy()[:10]  # First 5 labels
            print(labels)
            break  # Exit after printing the first batch

class CheckModelModeCallback(tf.keras.callbacks.Callback):
    def on_test_begin(self, logs=None):
        mode = 'Training' if self.model.training else 'Evaluation'
        print(f"Model mode at on_test_begin: {mode}")

        # Log as text using WandB Table
        table = wandb.Table(columns=["Phase", "Mode"])
        table.add_data("on_test_begin", mode)
        wandb.log({"model_mode": table})

    def on_test_end(self, logs=None):
        mode = 'Training' if self.model.training else 'Evaluation'
        print(f"Model mode at on_test_end: {mode}")

        # Log as text using WandB Table
        table = wandb.Table(columns=["Phase", "Mode"])
        table.add_data("on_test_end", mode)
        wandb.log({"model_mode": table})

class WandbCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        dict_keys = logs.keys()
        keys = list(dict_keys)
        logging.info("...Training: end of batch {}; got log keys: {}".format(batch, keys))
            
        wandb.log({
            "train_mae": logs['mean_absolute_error'], 
            "train_loss": logs['loss']
        })

    # def on_epoch_end(self, epoch, logs=None):
    #     dict_keys = logs.keys()
    #     keys = list(dict_keys)
    #     logging.info("...Training: start of epoch {}; got log keys: {}".format(epoch, keys))

    #     wandb.log({
    #         "val_mae": logs['val_mean_absolute_error'],
    #         "val_loss": logs['val_loss']
    #     })

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

    model_keras.add(tf.keras.layers.GlobalAveragePooling2D())
    model_keras.add(tf.keras.layers.BatchNormalization())
    model_keras.add(tf.keras.layers.Dropout(0.3))
    model_keras.add(tf.keras.layers.Dense(7, activation='linear'))
    
    ### DEFINE MODEL
    # def scheduler(epoch, lr):
    #     if epoch < 400:
    #         return 1e-4
    #     elif epoch < 800 :
    #         return 1e-5
    #     else :
    #         return 1e-5
        
    # layers_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/" + model_name,
    monitor='val_loss',
    verbose=config.verbose,
    save_best_only=True,
    mode='min',
    save_freq='epoch',
    initial_value_threshold=None
)
    model_keras.load_weights("/home/lecomte/AstroSpikes/2024-nc-hackathon-AstroSpikes/model_20241203_1634_.keras")
    model_keras.compile(loss=PoseEstimationLoss(beta = 5),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                        metrics=['mean_absolute_error'])
    
logging.info(model_keras.summary())
logging.info("Training model " + model_name)

callbacks = [checkpoint_callback, WandbCallback(), PrintLabelsCallback()] if  log_wandb else [checkpoint_callback] 

### TRAINING LOOP TO FIND BEST BETA VALUE

beta_values = [5,10,15,20,25,30]
best_beta = None
lowest_val_loss = float('inf')

for beta in beta_values:
    if log_wandb:
        wandb.init(# set the wandb project where this run will be logged
        project="mobilenet-astrospikes",
        name=config.transformation.method+"_beta_"+str(beta),
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(config))
    logging.info(f"Training with beta = {beta}")
    loss = PoseEstimationLoss(beta=beta)
    # Use the loss function with the current beta value
    model_keras.compile(loss=loss,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                        metrics=[config.metrics[0]])
    
    # Train the model
    history = model_keras.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              callbacks=callbacks, verbose=2, shuffle=True, validation_data=test_dataset)

    # Update the best beta value and val loss if needed
    val_loss = min(history.history["val_loss"])
    logging.info(f"Beta = {beta}, Best Validation Loss = {val_loss}")
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        best_beta = beta
    if log_wandb:
        wandb.finish()
logging.info(f"Best beta found: {best_beta} with validation loss {lowest_val_loss}")

# Train the model with the best beta value
loss = PoseEstimationLoss(beta=best_beta)
model_keras.compile(loss=loss,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                    metrics=[config.metrics[0]])
if log_wandb:
    wandb.init(# set the wandb project where this run will be logged
    project="mobilenet-astrospikes",
    name=config.transformation.method+"_final",
    # track hyperparameters and run metadata
    config=OmegaConf.to_container(config))
history = model_keras.fit(train_dataset, epochs=100, batch_size=BATCH_SIZE,
                          callbacks=callbacks, verbose=2, shuffle=True, validation_data=test_dataset)




if log_wandb :
    # Log the final model training to WandB
    wandb.log({"final_best_beta": best_beta, "final_val_loss": min(history.history["val_loss"])})
    wandb.finish()

# Save the final best model
model_keras.save(model_name)

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

json.dump(history, open("history_model.json", 'w'), default=set_default)
logging.info("Training over")