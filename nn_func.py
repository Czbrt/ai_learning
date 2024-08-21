import numpy as np
import ml_func as mlf


def dense(A_in, W, B, g):
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out


class Activation:
    def __init__(self, activation):
        self.activation = activation

    def forward(self, Z):
        if self.activation == 'relu':
            return mlf.relu(Z)
        elif self.activation == 'sigmoid':
            return mlf.sigmoid(Z)
        elif self.activation == 'tanh':
            return mlf.tanh(Z)
        else:
            return Z
