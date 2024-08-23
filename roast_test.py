import numpy as np
import tensor_nn as tn
import ml_func as mlf
import matplotlib.pyplot as plt

# 生成温度和时间的特征矩阵 (10组数据)
features = np.array([
    [180, 15],  # 成功
    [200, 10],  # 成功
    [220, 8],  # 成功
    [250, 5],  # 失败（温度过高）
    [160, 25],  # 成功
    [230, 7],  # 失败（温度过高）
    [170, 20],  # 成功
    [210, 12],  # 成功
    [240, 6],  # 失败（温度过高）
    [150, 30],  # 失败（温度不够）
])

features_normal = mlf.z_score_normalization(features)

# 生成相应的目标值数组 (1表示成功，0表示失败)
target = np.array([[1], [1], [1], [0], [1], [0], [1], [1], [0], [0]])

model = tn.MyModel()

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(features_normal, target, epochs=1000, verbose=0)

test = np.array([
    [200, 15],
    [220, 30],
    [1000, 100],
    [0, 0],
])

test_normal = mlf.z_score_normalization(test)
# 测试
predicted = model.predict(test_normal)
print(predicted)


# 生成数据
X = np.linspace(-1, 1, 100).reshape(-1, 1)  # 变为二维数组
y = X ** 2

quadratic_model = tn.MyModel()

# 编译模型
quadratic_model.compile(optimizer='adam', loss='mse')

# 训练模型
quadratic_model.fit(X, y, epochs=1000, verbose=0)

# 测试
predicted = quadratic_model.predict(X)

# 可视化
plt.scatter(X, y, label='True')
plt.plot(X, predicted, label='Predicted', color='red')
plt.legend()
plt.show()
