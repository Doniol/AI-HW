import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    inputs = keras.Input(shape=(784,))
    img_inputs = keras.Input(shape=(28, 28, 3))
    inputs.shape
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

