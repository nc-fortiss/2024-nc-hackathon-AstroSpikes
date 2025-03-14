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

if __name__ == '__main__':
    # loading omegaconf
    config_path = "configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    # initialize wandb
    run = wandb.init(project="mobilenet-astrospikes",
                     name=cfg.data.transformation + "_simple_branch",
                     config=OmegaConf.to_container(cfg, resolve=True),
                     mode=cfg.wandb
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

    # Wandb Metric Logger
    wml = WandbMetricsLogger(log_freq='batch')

    # dataset creation, saving the training files list and validation files list
    data_creation = CreateDF(cfg=cfg)
    data_creation.save_file(checkpoint_dir)
    train_df, val_df = data_creation()
    train_dataset = ImageDataLoader(cfg=cfg, df=train_df)
    val_dataset = ImageDataLoader(cfg=cfg, df=val_df)

    # Training Loop
    model.compile(loss=PoseEstimationLoss(),
                  optimizer=keras.optimizers.Adam(learning_rate=cfg.training.lr),
                  metrics={"position_output": "mse", "orientation_output": "mse"})

    model.fit(train_dataset,
              epochs=cfg.training.num_epochs,
              batch_size=cfg.training.batch_size,
              callbacks=[checkpoint_callback, wml],
              validation_data=val_dataset)

    run.finish()
