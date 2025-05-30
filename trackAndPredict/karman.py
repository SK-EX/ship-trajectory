import numpy as np


#定义非线性状态转移函数 x = [cx,cy,vx, vy] , dt = fps(video) / 1000是检测到当前物体的时间间隔，向量形式


#定义非线性观测函数x = [cx, cy, vx, vy]
def observation_function(x):

    #假设观测向量z = [cx, cy, ]
    return np.array([x[0], x[1]])

def state_transition_jacobian(x, dt):

    A = np.array([
        [1, 0, dt, 0],

        [0, 1, 0, dt],

        [0.1 * np.cos(x[0]), 0, 1, 0],

        [0, -0.1 * np.sin(x[1]), 0, 1]
    ])

    return A

def observation_jacobian(x):

    H = np.array([

        [1, 0, 0, 0],

        [0, 1, 0, 0]
    ])

    return H

class ExtendKalmanFilter:

    def __init__(self, state_dim, measure_dim):

        self.state_dim = state_dim

        self.measure_dim = measure_dim

        self.x = np.zeros(state_dim)

        self.P = np.eye(state_dim)

        self.Q = np.eye(state_dim) * 0.01

        self.R = np.eye(measure_dim) * 0.01

    def predict(self, dt):

        self.x = self.state_transition_function(self.x, dt)

        A = state_transition_jacobian(self.x, dt)

        self.P = A @ self.P @ A.T + self.Q

    #z为观测值
    def update(self, z):

        H = observation_jacobian(self.x)

        y = z - observation_function(self.x)

        S = H @ self.P @ H.T + self.R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def state_transition_function(self, x, dt):
        x_new = np.zeros_like(x)

        x_new[0] = x[0] + x[2] * dt

        x_new[1] = x[1] + x[3] * dt

        x_new[2] = x[2] + 0.1 * np.sin(x[0])

        x_new[3] = x[3] + 0.1 * np.cos(x[1])

        return x_new


def mainFunction(original_position, trajectory):
    # 初始化卡尔曼滤波器
    kf = ExtendKalmanFilter(state_dim = 4, measure_dim = 2)

    #这里的初始状态取自的original_position = trajectories[i][0]
    kf.x = original_position

    # 假设初始协方差矩阵为单位矩阵
    kf.P = np.eye(4)

    # 假设观测噪声协方差矩阵为单位矩阵
    kf.R = np.eye(2)

    # 假设过程噪声协方差矩阵为单位矩阵
    kf.Q = np.eye(4)

    # 假设观测值为 [cx, cy]

    z = trajectory

    #存储真实值，观测值，估计值
    real_values = []

    measure_values = []

    estimate_values = []





