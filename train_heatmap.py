import argparse
import functools
import glob
import logging
import os
from datetime import datetime
from pathlib import Path
import keras
import pandas as pd
import tensorflow as tf
import wandb
from omegaconf import OmegaConf
from wandb.integration.keras import WandbMetricsLogger

# Import your project's modules
from src.dataloaders.spades import create_dataset
from src.losses.poseloss import mpkpe_regression, position_mse_loss
from src.utils.kdsnt import mpkpe_heatmap, heatmap_kl_l2_loss, heatmap_kl_loss, heatmap_l2_loss
from src.models.mobilenet_heatmap import mobilenet_heatmap_1pass as mobilenet_heatmap_model
# from src.models.mobilenet import mobilenet_heatmap_akida as mobilenet_heatmap_model

# It's good practice to use a logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    import random
    import numpy as np
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_environment(cfg):
    """Sets up environment variables and creates directories."""
    os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = cfg.model.environment.cnn2snn_target_akida_version

    # Create a unique run directory with a timestamp for safety
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"{cfg.paths.checkpoint_dir}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Checkpoint and log directory created at: {run_dir}")

    return run_dir


def initialize_wandb(cfg, run_dir, config_path):
    """Initializes and configures a new WandB run."""
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=run_dir,
        # name=f"{cfg.id}-{datetime.now().strftime('%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.status
    )
    logging.info(f"WandB run initialized with name: {run.name}")

    code_artifact = wandb.Artifact(
        name=f"source-code-{wandb.run.id}", type="code"
    )

    paths_to_log = set()

    paths_to_log.add(str(Path(config_path)))

    # This part remains the same
    for path in glob.glob('src/**/*.py', recursive=True):
        paths_to_log.add(path)
    paths_to_log.add(str(Path('train_heatmap.py')))

    print(f"Logging {len(paths_to_log)} source files to W&B Artifact...")
    for unique_path in sorted(list(paths_to_log)):
        # This will now work because `unique_path` for the config
        # will be the full, correct path like 'configs/my_config.yaml'
        code_artifact.add_file(unique_path)

    wandb.log_artifact(code_artifact)

    return run


def get_datasets(cfg):
    """Loads and creates train and validation datasets."""
    logging.info(f"Loading training data from: {cfg.paths.train_data}")
    train_df = pd.read_csv(cfg.paths.train_data)
    val_df = pd.read_csv(cfg.paths.val_data)

    train_data = {col: train_df[col].values for col in train_df.columns}
    val_data = {col: val_df[col].values for col in val_df.columns}

    train_dataset = create_dataset(
        train_data,
        batch_size=cfg.training.batch_size,
        input_size=cfg.data.input_size[:2],
        is_training=True,
        heatmap=cfg.model.heatmap,
        cache_dir=None
    )
    val_dataset = create_dataset(
        val_data,
        batch_size=cfg.training.batch_size,
        input_size=cfg.data.input_size[:2],
        is_training=False,
        heatmap=cfg.model.heatmap,
        cache_dir=None
    )

    train_size = len(train_data[next(iter(train_data))])
    val_size = len(val_data[next(iter(val_data))])

    logging.info(f"Datasets created. Train size: {train_size}, Validation size: {val_size}")
    return train_dataset, val_dataset, train_size, val_size


def get_compiler_args(cfg):
    """Constructs loss, optimizer, and metrics from config."""
    # --- Learning Rate Schedule ---
    # lr_cfg = cfg.training.optimizer.learning_rate
    # if lr_cfg.schedule == "ExponentialDecay":
    #     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #         lr_cfg.initial_lr,
    #         decay_steps=lr_cfg.decay_steps,
    #         decay_rate=lr_cfg.decay_rate,
    #         staircase=lr_cfg.staircase
    #     )
    # else:
    #     raise ValueError(f"Unsupported learning rate schedule: {lr_cfg.schedule}")

    # --- Optimizer ---
    opt_cfg = cfg.training.optimizer
    if opt_cfg.name == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_cfg.learning_rate.initial_lr)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    # --- Loss Function ---
    # loss_cfg = cfg.training.loss
    if cfg.model.heatmap:
        # L2_WEIGHT = 0.5
        # loss_fn = functools.partial(heatmap_kl_l2_loss, lambda_l2=L2_WEIGHT)
        # loss_fn.__name__ = f"kl_l2_loss_lambda_{L2_WEIGHT}"
        loss_fn = heatmap_l2_loss
        loss_fn.__name__ = f"heatmap_l2_loss"
        # loss_fn_name = "KLDivergence"
        losses = {"heatmap_output": loss_fn}
    else:
        losses = {"heatmap_output": position_mse_loss}

    # --- Metrics ---
    if cfg.model.heatmap:
        metrics = {"heatmap_output": mpkpe_heatmap}
    else:
        metrics = {"heatmap_output": mpkpe_regression}

    return {"optimizer": optimizer, "loss": losses, "metrics": metrics}


def build_model(cfg, strategy=None):
    """Builds and compiles the Keras model."""
    compiler_args = get_compiler_args(cfg)

    def create_and_compile():
        model = mobilenet_heatmap_model(
            input_size=list(cfg.data.input_size),
            num_keypoints=cfg.model.num_keypoints,
        )
        model.compile(**compiler_args)
        return model

    if strategy:
        with strategy.scope():
            model = create_and_compile()
            logging.info("Model built with MirroredStrategy.")
    else:
        model = create_and_compile()
        logging.info("Model built on a single device.")

    # Build the model to inspect summary
    model.build(input_shape=(None, *cfg.data.input_size))
    # Log model summary to WandB
    s = []
    model.summary(print_fn=lambda x: s.append(x))
    wandb.log({"model_summary": "\n".join(s)})

    return model


def get_callbacks(cfg, run_dir):
    """Creates the list of callbacks for model.fit(). (Local Saving Only)"""

    callbacks = [
        # Logs metrics to W&B, does not save models
        WandbMetricsLogger(log_freq=cfg.wandb.log_freq),

        # Saves best model locally based on validation loss
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, 'checkpoints/{epoch:02d}-{val_loss:.4f}.keras'),
            monitor=cfg.checkpointing.monitor,
            mode=cfg.checkpointing.mode,
            save_best_only=True
        ),
    ]

    logging.info("Configured for local checkpointing only. W&B model checkpointing disabled.")
    return callbacks


def main(config_path):
    """Main training pipeline."""
    # Load configuration
    cfg = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)
    # Setup environment and directories
    run_dir = setup_environment(cfg)

    # Initialize WandB
    run = initialize_wandb(cfg, run_dir, config_path=config_path)

    # Load data
    train_dataset, val_dataset, train_size, val_size = get_datasets(cfg)

    steps_per_epoch = train_size // cfg.training.batch_size
    validation_steps = val_size // cfg.training.batch_size

    # --- START OF CORRECTION ---

    # 1. Setup distributed training strategy if enabled
    if cfg.training.distributed:
        strategy = tf.distribute.MirroredStrategy()
        logging.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices.")
    else:

        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = build_model(cfg)

    # --- END OF CORRECTION ---

    # Get callbacks
    callbacks = get_callbacks(cfg, run_dir)

    # Start training
    logging.info("Starting model training...")
    model.fit(
        train_dataset,
        epochs=cfg.training.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        use_multiprocessing=cfg.training.use_multiprocessing,
        verbose=cfg.training.verbose
    )

    logging.info("Training finished.")
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a keypoint detection model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/heatmap_dsnt.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    main(args.config)
