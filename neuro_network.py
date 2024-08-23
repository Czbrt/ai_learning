import numpy as np
import ml_func as mlf


class Dense:
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.W = None
        self.b = None

        self.Z = None  # Initialize self.Z
        self.A_in = None  # Initialize self.A_in
        self.A_out = None  # Initialize self.A_out

    def initialize(self, input_dim):
        # 权重初始化
        self.W = np.random.randn(input_dim, self.units) * np.sqrt(2 / input_dim)
        self.b = np.zeros((1, self.units))

    def forward(self, A_in):
        self.A_in = A_in  # Save input for backpropagation
        self.Z = np.dot(A_in, self.W) + self.b  # Store Z for backward propagation
        if self.activation == 'relu':
            self.A_out = mlf.relu(self.Z)
        elif self.activation == 'sigmoid':
            self.A_out = mlf.sigmoid(self.Z)
        elif self.activation == 'tanh':
            self.A_out = mlf.tanh(self.Z)
        else:
            self.A_out = self.Z
        return self.A_out

    def backward(self, dA_out, learning_rate):
        if self.activation == 'relu':
            dZ = dA_out * (self.Z > 0)
        elif self.activation == 'sigmoid':
            sig = mlf.sigmoid(self.Z)
            dZ = dA_out * sig * (1 - sig)
        elif self.activation == 'tanh':
            dZ = dA_out * (1 - np.power(mlf.tanh(self.Z), 2))
        else:
            dZ = dA_out

        dW = np.dot(self.A_in.T, dZ)  # Ensure the shape matches
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_in = np.dot(dZ, self.W.T)

        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_in


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

    def backward(self, dA, learning_rate):
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    def fit(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            # Forward propagation
            A_out = self.forward(X)
            # Compute loss
            loss = mlf.binary_crossentropy(y, A_out)
            if i % 100 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {loss:.2f}")
            # Backward propagation
            dA = -(y / A_out) + (1 - y) / (1 - A_out)
            self.backward(dA, learning_rate)
