import os
from datetime import datetime

import wandb
from omegaconf import OmegaConf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger

from src.dataloaders.spades import ImageDataLoader, CreateDF
from src.losses.poseloss import PoseEstimationLoss
from src.models.mobilenet import MobilenetModel


def train(config, train_data, val_data, model, optimizer_fn, loss_fn, callbacks_list):
    model.compile(loss=loss_fn,
                  optimizer=optimizer_fn,
                  metrics={"position_output": "mse", "orientation_output": "mse"}
                  )

    model.fit(train_data,
              epochs=config.training.num_epochs,
              batch_size=config.training.batch_size,
              callbacks=callbacks_list,
              validation_data=val_data)

    return


if __name__ == '__main__':
    # initialize wandb with your project name and optionally with configutations.

    wandb.login(key='')

    config_path = "configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(project="mobilenet-astrospikes",
                     name=cfg.data.transformation + "_simple_branch",
                     config=OmegaConf.to_container(cfg, resolve=True),
                     mode='offline'
                     )
    # if not cfg.log.wandb:
    wandb.log({'config': str(wandb.config)})

    # Initialize model.
    input_shape = list(cfg.data.input_size)  # Convert ListConfig to a standard list
    model = MobilenetModel(input_size=input_shape, pretrained=cfg.model.pretrained)
    model.build(input_shape=(None, *input_shape))  # None is for batch size

    # model = make_model(input_size=cfg.data.input_size, pretrained=cfg.model.pretrained)
    wandb.log({"model_summary": model.summary()})

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam(learning_rate=cfg.training.lr)
    # Instantiate a loss function.
    pose_loss = PoseEstimationLoss()

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
        mode='min'
    )

    wml = WandbMetricsLogger(log_freq='batch')

    data_creation = CreateDF(cfg=cfg)
    data_creation.save_file(checkpoint_dir)
    train_df, val_df = data_creation()
    train_dataset = ImageDataLoader(cfg=cfg, df=train_df)
    val_dataset = ImageDataLoader(cfg=cfg, df=val_df)

    train(cfg,
          train_dataset,
          val_dataset,
          model,
          optimizer,
          pose_loss,
          callbacks_list=[checkpoint_callback, wml])

    run.finish()
