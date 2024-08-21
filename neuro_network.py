import numpy as np
import ml_func as mlf


class Dense:
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.W = None
        self.b = None

    def initialize(self, input_dim):
        # 权重初始化
        self.W = np.random.randn(input_dim, self.units) * 0.01
        self.b = np.zeros((1, self.units))

    def forward(self, A_in):
        A_in = A_in
        Z = np.dot(A_in, self.W) + self.b
        if self.activation == 'relu':
            A_out = mlf.relu(Z)
        elif self.activation == 'sigmoid':
            A_out = mlf.sigmoid(Z)
        elif self.activation == 'tanh':
            A_out = mlf.tanh(Z)
        else:
            A_out = Z
        return A_out


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def initialize(self, input_dim):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.initialize(input_dim)
                input_dim = layer.units

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
