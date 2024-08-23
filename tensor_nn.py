import tensorflow as tf


# 自定义 Dense 层
class Dense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_bias=True):
        super(Dense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias

        self.bias = None
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="kernel")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            x = x + self.bias
        if self.activation:
            x = self.activation(x)
        return x


# 定义模型
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(units=10, activation=tf.keras.activations.relu)
        self.dense2 = Dense(units=5, activation=tf.keras.activations.relu)
        self.dense3 = Dense(units=1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
