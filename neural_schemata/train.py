"""Simple training pipeline implemented in Tensorflow."""

import argparse

import keras
import tensorflow as tf


class MyModel(keras.Model):  # pylint: disable=abstract-method
    """Simple Keras model."""

    def __init__(self) -> None:
        """Initializes instance of the class."""
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pylint: disable=arguments-differ
        """Evaluates the model."""
        x = self.dense1(inputs)
        return self.dense2(x)


def parse_command_line() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def run() -> None:
    """Entry point of the script."""
    _ = parse_command_line()
    model = MyModel()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


if __name__ == "__main__":
    run()
