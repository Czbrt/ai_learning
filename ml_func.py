import numpy as np


def lin_comb(X, W, b):
    """
    计算线性组合：z = XW + b

    参数：
    X -- 输入特征矩阵，形状为 (m, n)
    W -- 权重矩阵，形状为 (n, 1)
    b -- 偏置项，标量

    返回：
    z -- 线性组合的结果，形状为 (m, 1)
    """
    z = np.dot(X, W) + b
    return z


def z_score_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normal = (X - mean) / std
    return X_normal


# classification
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# Linear Regression
def relu(Z):
    return np.maximum(0, Z)


def tanh(Z):
    return np.tanh(Z)


def binary_crossentropy(y_true, y_pred):
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
