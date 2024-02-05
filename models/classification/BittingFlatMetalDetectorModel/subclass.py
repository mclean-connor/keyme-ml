import tensorflow as tf
from tensorflow import keras  # type: ignore
from models.classification.BittingFlatMetalDetectorModel import config


class BittingFlatMetalDetectorSubclass(keras.Model):
    def __init__(self, num_classes, data_format="NHWC"):
        super(BittingFlatMetalDetectorSubclass, self).__init__()
        self.num_classes = num_classes
        self.data_format = data_format

        self.conv1 = keras.layers.Conv2D(
            32,
            (3, 3),
            strides=(2, 2),
            padding="same",
            data_format=data_format,
            activation="relu",
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(0.5)

        self.conv2 = keras.layers.Conv2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
            data_format=data_format,
            activation="relu",
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.dropout2 = keras.layers.Dropout(0.5)

        if config.use_third_convolutional_block:
            self.conv3 = keras.layers.Conv2D(
                128,
                (3, 3),
                strides=(2, 2),
                padding="same",
                data_format=data_format,
                activation="relu",
            )
            self.bn3 = keras.layers.BatchNormalization()
            self.dropout3 = keras.layers.Dropout(0.5)

        self.global_avg_pool = keras.layers.GlobalAveragePooling2D(
            data_format=data_format
        )
        self.classifier = keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = tf.divide(inputs, 255.0)
        if self.data_format == "NCHW":
            x = tf.transpose(x, [0, 3, 1, 2])

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        if config.use_third_convolutional_block:
            x = self.conv3(x)
            x = self.bn3(x, training=training)
            x = self.dropout3(x, training=training)

        x = self.global_avg_pool(x)
        probs = self.classifier(x)

        return probs
