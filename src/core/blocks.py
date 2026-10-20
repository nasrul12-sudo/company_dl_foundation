import tensorflow as tf
from keras import layers

class HiddenBlock(layers.Layer):
    def __init__(self, units, activation="relu", dropout=0.0, use_bn=True, **kw):
        super().__init__(**kw)
        self.dense = layers.Dense(units, activation=activation)
        self.bn = layers.BatchNormalization() if use_bn else None
        self.drop = layers.Dropout(dropout) if dropout > 0 else None

    def call(self, x, training=False):
        x = self.dense(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.drop is not None:
            x = self.drop(x, training=training)
        return x

class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout=0.0, **kw):
        super().__init__(**kw)
        self.fc1 = layers.Dense(units, activation="relu")
        self.fc2 = layers.Dense(units)
        self.drop = layers.Dropout(dropout) if dropout > 0 else None

    def call(self, x, training=False):
        h = self.fc1(x)
        h = self.fc2(h)

        if self.drop is not None:
            h = self.drop(h, training=training)
        return tf.nn.relu(x + h) if x.shape[-1] == h.shape[-1] else tf.nn.relu(h)

    