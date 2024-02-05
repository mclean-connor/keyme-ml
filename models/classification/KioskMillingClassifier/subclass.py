import tensorflow as tf
import keras from tensorflow # type: ignore

class KioskMillingClassifierSubclass(keras.Model):
    def __init__(self, y_to_label, data_format='NHWC'):
        super(KioskMillingClassifierSubclass, self).__init__()
        self.y_to_label = y_to_label
        self.num_classes = len(y_to_label)
        self.data_format = data_format

        self.conv1 = keras.layers.Conv2D(32, 3, strides=2, padding='same', data_format=data_format, activation='relu')
        self.bn1 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(0.5)

        self.conv2 = keras.layers.Conv2D(64, 3, strides=2, padding='same', data_format=data_format, activation='relu')
        self.bn2 = keras.layers.BatchNormalization()
        self.dropout2 = keras.layers.Dropout(0.5)

        self.conv3 = keras.layers.Conv2D(128, 3, strides=2, padding='same', data_format=data_format, activation='relu')
        self.bn3 = keras.layers.BatchNormalization()
        self.dropout3 = keras.layers.Dropout(0.5)

        self.conv4 = keras.layers.Conv2D(128, 3, padding='same', data_format=data_format, activation='relu')
        self.bn4 = keras.layers.BatchNormalization()
        self.dropout4 = keras.layers.Dropout(0.5)

        self.global_avg_pool = keras.layers.GlobalAveragePooling2D(data_format=data_format)
        self.classifier = keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=None):
        if self.data_format == 'NCHW':
            x = tf.transpose(inputs, [0, 3, 1, 2])  # Convert 'NHWC' to 'NCHW'
        else:
            x = inputs
        x = tf.divide(x, 255.0)  # Scale the input

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)

        x = self.global_avg_pool(x)

        return self.classifier(x)
