import numpy as np


# 假设我们有一个简单的目标函数，我们希望通过调整参数来最小化它
def objective_function(w, b):
    return np.sum((w * X + b - Y) ** 2)


# 随机生成一些数据来模拟输入和输出
np.random.seed(42)
N = 100
X = np.random.randn(N, 1)
Y = 2 * X + 3 + np.random.randn(N, 1) * 0.1  # 真实参数是 w=2, b=3

# 初始化参数
w = np.random.randn(1)
b = np.random.randn(1)

# 学习率
learning_rate = 0.01


# 梯度下降参数更新
def stochastic_gradient_descent(X, Y, w, b, learning_rate, iterations):
    for i in range(iterations):
        # 随机选择一个样本
        idx = np.random.randint(0, len(X))
        x_i = X[idx]
        y_i = Y[idx]

        # 计算梯度
        dw = (2 * x_i * (w * x_i + b - y_i) * x_i).sum()
        db = (2 * (w * x_i + b - y_i)).sum()

        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

        loss = objective_function(w, b)

        # 打印每100次迭代的损失
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return w, b


# 运行随机梯度下降
w_opt, b_opt = stochastic_gradient_descent(X, Y, w, b, learning_rate, 1000)

print(f"Optimized parameters: w = {w_opt}, b = {b_opt}")