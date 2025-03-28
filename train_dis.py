import os
import tensorflow as tf
from datetime import datetime
import pandas as pd
import wandb
from omegaconf import OmegaConf
from tensorflow.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger
import glob

from src.dataloaders.spades import create_dataset
from src.losses.poseloss import geodesic_dist, position_mse_loss, quat_rel_angle
from src.models.mobilenet import MobilenetModel

if __name__ == '__main__':
    # loading omegaconf
    config_path = "configs/distributed.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    exp_folder = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    checkpoint_dir = str(os.path.join(cfg.root.checkpoint, exp_folder))
    os.makedirs(checkpoint_dir, exist_ok=True)
    logdir = str(os.path.join(checkpoint_dir, 'logs'))
    # wandb.tensorboard.patch(root_logdir=logdir)

    # initialize wandb
    run = wandb.init(project="mobilenet-astrospikes",
                     dir=checkpoint_dir,
                     name=cfg.wandb.exp_id,
                     config=OmegaConf.to_container(cfg, resolve=True),
                     mode=cfg.wandb.status,
                     )
    wandb.log({'config': str(wandb.config)})
    if cfg.wandb.status == 'online':
        code = wandb.Artifact('project-source', type='code')
        for path in glob.glob('**/*.py', recursive=True):
            code.add_file(path)
        for path in glob.glob('**/*.yaml', recursive=True):
            code.add_file(path)
        wandb.run.use_artifact(code)

    # Setup WandbModelCheckpoint
    model_name = "model_{epoch:02d}_{val_loss:.4f}.keras"
    checkpoint_callback = ModelCheckpoint(
        str(os.path.join(checkpoint_dir, model_name)),
        monitor='val_loss',
        verbose=cfg.training.verbose,
        save_best_only=True,
        mode='min')

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=logdir,
    #     update_freq='batch')

    # Wandb Metric Logger
    wml = WandbMetricsLogger(log_freq='batch')

    # dataset creation, saving the training files list and validation files list
    train_df = pd.read_csv(os.path.join(cfg.root.data_out, 'train.csv'))
    train_data = {col: train_df[col].values for col in train_df.columns}

    val_df = pd.read_csv(os.path.join(cfg.root.data_out, 'val.csv'))
    val_data = {col: val_df[col].values for col in val_df.columns}

    train_dataset = create_dataset(train_data,
                                   batch_size=cfg.training.batch_size,
                                   input_size=cfg.data.input_size[:2],
                                   is_training=True,
                                   cache_dir=None)
    # train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality(), reshuffle_each_iteration=True)

    val_dataset = create_dataset(val_data,
                                 batch_size=cfg.training.batch_size,
                                 input_size=cfg.data.input_size[:2],
                                 is_training=False,
                                 cache_dir=None)
    # val_dataset = val_dataset.shuffle(buffer_size=val_dataset.cardinality(), reshuffle_each_iteration=True)

    initial_learning_rate = cfg.training.lr
    decay_steps = 5e4  # Adjust based on your dataset size and epochs
    decay_rate = 0.96  # Typical value

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    # Define loss functions
    losses = {
        "position": position_mse_loss,
        "orientation": geodesic_dist
    }

    # Assign different importance to losses
    loss_weights_dict = {
        "position": 0.2,
        "orientation": 0.8
    }

    metrics_dict = {
        "position": ["mae"],  # Two metrics for position
        "orientation": quat_rel_angle  # Only MSE for orientation
    }

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        # Initialize model.
        model = MobilenetModel(input_size=list(cfg.data.input_size), pretrained=cfg.model.pretrained)
        # model.build(input_shape=(None, *list(cfg.data.input_size)))  # None is for batch size
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # Training Loop
        model.compile(loss=losses,
                      loss_weights=loss_weights_dict,
                      optimizer=optimizer,
                      metrics=metrics_dict
                      )

    wandb.log({"model_summary": model.summary()})
    print(OmegaConf.to_yaml(cfg))
    print('Exp_ID:', exp_folder)

    # model.compile(loss=BetaLoss(), optimizer=optimizer)  # check beta = {2, 5, 10, 20}
    steps_per_epoch = len(train_data[next(iter(train_data))]) // cfg.training.batch_size  # Calculate steps per epoch
    validation_steps = len(val_data[next(iter(val_data))]) // cfg.training.batch_size

    model.fit(train_dataset,
              epochs=cfg.training.num_epochs,
              batch_size=cfg.training.batch_size,
              steps_per_epoch=steps_per_epoch,
              callbacks=[checkpoint_callback, wml],
              validation_data=val_dataset,
              validation_steps=validation_steps,
              use_multiprocessing=True)

    run.finish()
