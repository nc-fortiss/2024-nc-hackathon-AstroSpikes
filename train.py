import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from tensorflow import keras
from wandb.integration.keras import WandbModelCheckpoint

import wandb
from src.dataloaders.spades import ImageDataLoader, createDF
from src.losses.poseloss import PoseEstimationLoss
from src.models.mobilenet import MyModel


@tf.function
def train_step(x, y, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss_value = loss_fn(y, preds)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def test_step(x, y, model, loss_fn):
    val_preds = model(x, training=False)
    loss_value = loss_fn(y, val_preds)
    return loss_value


def train(train_dataset, val_dataset, model, optimizer, loss_fn, epochs=10):
    best_val_acc = 100.0
    checkpoint_dir = str(os.path.join(cfg.root.checkpoint, cfg.id, datetime.now().strftime("%Y%m%d_%H%M%S")))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(cfg.root.checkpoint, exist_ok=True)

    # Setup WandbModelCheckpoint
    model_name = "model_{epoch:02d}_{val_loss:.4f}.keras"
    checkpoint_callback = WandbModelCheckpoint(
        str(os.path.join(checkpoint_dir, model_name)),
        monitor='val_loss',
        verbose=cfg.training.verbose,
        save_best_only=True,
        mode='min'
    )

    for epoch in range(epochs):

        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train,
                                    model, optimizer,
                                    loss_fn)
            wandb.log({'loss': float(loss_value)})
            train_loss.append(float(loss_value))

        # Run a validation loop at the end of each epoch
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val,
                                       model, loss_fn
                                       )
            wandb.log({'val_loss': float(val_loss_value)})
            val_loss.append(float(val_loss_value))

        # ⭐: log metrics using wandb.log
        wandb.log({'epochs': epoch})

        checkpoint_callback.set_model(model)
        if np.mean(val_loss) < best_val_acc:
            best_val_acc = np.mean(val_loss)
            checkpoint_callback.on_epoch_end(epoch, logs={'val_loss': np.mean(val_loss)})


if __name__ == '__main__':
    # initialize wandb with your project name and optionally with configutations.

    wandb.login(key='')

    config_path = "configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    if not cfg.log.wandb:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project="mobilenet-astrospikes",
                     name=cfg.data.transformation + "_simple_branch",
                     config=OmegaConf.to_container(cfg, resolve=True)
                     )
    wandb.log({'config': str(wandb.config)})

    # Initialize model.
    input_shape = list(cfg.data.input_size)  # Convert ListConfig to a standard list
    model = MyModel(input_size=input_shape, pretrained=cfg.model.pretrained)
    model.build(input_shape=(None, *input_shape))  # None is for batch size

    # model = make_model(input_size=cfg.data.input_size, pretrained=cfg.model.pretrained)
    wandb.log({"model_summary": model.summary()})

    # Instantiate an optimizer to train the model.
    optimizer = keras.optimizers.Adam(learning_rate=cfg.training.lr)
    # Instantiate a loss function.
    loss_fn = PoseEstimationLoss(beta=5)

    # Load the dataset
    data_creation = createDF(cfg=cfg)
    train_df, val_df = data_creation()
    train_dataset = ImageDataLoader(cfg=cfg, df=train_df)
    val_dataset = ImageDataLoader(cfg=cfg, df=val_df)

    train(train_dataset,
          val_dataset,
          model,
          optimizer,
          loss_fn,
          epochs=cfg.training.num_epochs
          )

    run.finish()
