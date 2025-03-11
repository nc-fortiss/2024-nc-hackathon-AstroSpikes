import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model

import wandb
from wandb.integration.keras import WandbMetricsLogger

import os

from DataLoading.image_loader import ImageDataLoader

#set up wandb
log_wandb = True

if log_wandb:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="mobilenet-astrospikes",
        entity="neuromorphic-research", 
        name="native_implementation",
        config={
            "epochs": 10,
            "learning_rate": 0.001,
            "architecture": "MobileNetV1",
        }
    )

epochs = wandb.config.epochs
learning_rate = wandb.config.learning_rate

# Define the layers
input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)

base_model = MobileNet(input_tensor=inputs, include_top=False, weights="imagenet")

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)

outputs = Dense(7, activation="linear")(x)  

# Define model
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss="mse",
              metrics=["mae"])

model.summary()

# load dataset
loader = ImageDataLoader(test=True)
train, test = loader()

# Train the model
model.fit(train, epochs=epochs, validation_data=test, callbacks = [WandbMetricsLogger(log_freq="batch")])

# Save the model
model.save("mobilenet_astrospikes_model.keras")