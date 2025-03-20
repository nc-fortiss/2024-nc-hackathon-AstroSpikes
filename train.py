import os
from datetime import datetime

import wandb
from omegaconf import OmegaConf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger

from src.dataloaders.spades import CreateDF, create_dataset
from src.losses.poseloss import PoseEstimationLoss
from src.models.mobilenet import MobilenetModel

if __name__ == '__main__':
    # loading omegaconf
    config_path = "configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    # initialize wandb
    run = wandb.init(project="mobilenet-astrospikes",
                     name=cfg.wandb.exp_id,
                     config=OmegaConf.to_container(cfg, resolve=True),
                     mode=cfg.wandb.status
                     )
    wandb.log({'config': str(wandb.config)})

    # Initialize model.
    input_shape = list(cfg.data.input_size)  # Convert ListConfig to a standard list
    model = MobilenetModel(input_size=input_shape, pretrained=cfg.model.pretrained)
    model.build(input_shape=(None, *input_shape))  # None is for batch size
    wandb.log({"model_summary": model.summary()})

    exp_folder = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    checkpoint_dir = str(os.path.join(cfg.root.checkpoint, exp_folder))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup WandbModelCheckpoint
    model_name = "model_{epoch:02d}_{val_loss:.4f}.keras"
    checkpoint_callback = ModelCheckpoint(
        str(os.path.join(checkpoint_dir, model_name)),
        monitor='val_loss',
        verbose=cfg.training.verbose,
        save_best_only=True,
        mode='min')

    logdir = "logs/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        update_freq='batch')

    # Wandb Metric Logger
    wml = WandbMetricsLogger(log_freq='batch')

    # dataset creation, saving the training files list and validation files list
    data_creation = CreateDF(cfg=cfg)
    data_creation.save_file(checkpoint_dir)
    train_df, val_df = data_creation()

    train_dataset = create_dataset(train_df, batch_size=cfg.training.batch_size,
                                   input_size=cfg.data.input_size[:2],
                                   is_training=True,
                                   cache_dir=None)
    # train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality(), reshuffle_each_iteration=True)
    val_dataset = create_dataset(val_df, batch_size=cfg.training.batch_size,
                                 input_size=cfg.data.input_size[:2],
                                 is_training=False,
                                 cache_dir=None)
    # val_dataset = val_dataset.shuffle(buffer_size=val_dataset.cardinality(), reshuffle_each_iteration=True)

    initial_learning_rate = cfg.training.lr
    decay_steps = 1e4  # Adjust based on your dataset size and epochs
    decay_rate = 0.96  # Typical value

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    # Training Loop
    model.compile(loss=PoseEstimationLoss(),
                  optimizer=optimizer)

    steps_per_epoch = len(train_df) // cfg.training.batch_size  # Calculate steps per epoch
    validation_steps = len(val_df) // cfg.training.batch_size

    model.fit(train_dataset,
              epochs=cfg.training.num_epochs,
              batch_size=cfg.training.batch_size,
              steps_per_epoch=steps_per_epoch,
              callbacks=[checkpoint_callback, wml, tensorboard_callback],
              validation_data=val_dataset,
              validation_steps=validation_steps)

    run.finish()
