file = open("result.txt",'r')

trajectories = []

for line in file:

    line = line.strip()

    trajectories.append(line)

tuple_1 = trajectories[0]

tuple_2 = trajectories[2]

import numpy as np

import ast  # 用于安全地解析字符串

from matplotlib import pyplot as plt


# 原始字符串
string = '[(321,423),(423,423),(433,423)......]'

# 解析字符串为 Python 列表
data = ast.literal_eval(tuple_1)

data_2 = ast.literal_eval(tuple_2)

# 转换为 numpy 数组
array_1 = np.array(data)

array_2 = np.array(data_2)

#用卡尔曼做实验

import karman

EKF = karman.ExtendKalmanFilter(state_dim= 4, measure_dim = 2)

EKF2 = karman.ExtendKalmanFilter(state_dim= 4, measure_dim = 2)

ekf_list = []

ekf_list2 = []

original_position1 = np.array([array_1[0][0],array_1[0][1], 0, 0])

original_position2 = np.array([array_2[0][0],array_2[0][1], 0, 0])

EKF.x = original_position1

EKF2.x = original_position2

z_x = np.array(array_1[:,0])

z_y = np.array(array_1[:,1])

z = np.array([z_x,z_y])

z = z.T


z_x2 = np.array(array_2[:,0])

z_y2 = np.array(array_2[:,1])

z2 = np.array([z_x2,z_y2])

z2 = z2.T

future_predictions1 = []

future_predictions2 = []

for item in range(len(z)):

    EKF.predict(1/25)

    EKF.update(z[item])

    ekf_list.append(EKF.x[:2])

    future_state = []

    state_predict = EKF.x

    for i in range(25):

        state_predict = EKF.state_transition_function(state_predict, 1/25)

        future_state.append(state_predict[:2])  # 保存未来坐标

    future_predictions1.append(future_state)


for item in range(len(z2)):

    EKF2.predict(1/25)

    EKF2.update(z2[item])

    ekf_list2.append(EKF2.x[:2])

    future_state = []

    state_predict = EKF2.x

    for i in range(25):

        state_predict = EKF2.state_transition_function(state_predict, 1/25)

        future_state.append(state_predict[:2])  # 保存未来坐标

    future_predictions2.append(future_state)


# plt.plot(array_2[:,0],array_2[:,1],'r',label = 'trajectory2')

#tutple数组to list
ekf_list = np.array(ekf_list)

ekf_list2 = np.array(ekf_list2)

future_predictions = np.array(future_predictions1)

future_predictions2 = np.array(future_predictions2)


first_dim = np.shape(future_predictions)[0]

first_dim2 = np.shape(future_predictions2)[0]

future_predictions = future_predictions.reshape([first_dim * 25,2])

future_predictions2 = future_predictions2.reshape([first_dim2 * 25,2])

print(array_1)

print(future_predictions)

print(future_predictions2)

#误差分析
N1  = np.shape(array_1)[0]

N2 = np.shape(array_2)[0]

MSE_x = 1 / N1 * np.sum((array_1[:,0] - ekf_list[:,0])**2)

MSE_y = 1 / N1 * np.sum((array_1[:,1] - ekf_list[:,1])**2)

#因为0.5秒后的轨迹应该与真是轨迹的12帧后面进行比较所以就用这个循环

MSE_x_future = []

MSE_y_future = []

temp1 = 0
temp2 = 0

for i in range(N2):

    for j in range(25):

        if i + j < N2:

           temp1 += (array_2[i + j,0] - future_predictions2[25 * i + j,0])**2

           temp2 += (array_2[i + j,1] - future_predictions2[25 * i + j,1])**2

    MSE_x_future.append(temp1 / 12)

    MSE_y_future.append(temp2 / 12)

    temp1 = 0
    temp2 = 0

MSE_x_future = np.array(MSE_x_future)
MSE_y_future = np.array(MSE_y_future)

print(MSE_x_future)
print(MSE_y_future)

plt.plot(MSE_x_future,'r',label = 'MSE_x_future')
plt.plot(MSE_y_future,'b',label = 'MSE_y_future')
plt.title(' MSE of 1s ')
plt.legend(['MSE_x_future','MSE_y_future'])
plt.show()

#plt.plot(array_1[:,0],array_1[:,1],'b', label = 'trajectory1')

plt.plot(array_2[:,0],array_2[:,1],'r',label = 'trajectory2')

#plt.plot(ekf_list[:,0],ekf_list[:,1],'r',label = 'ekf')

plt.plot(ekf_list2[:,0],ekf_list2[:,1],'b',label = 'ekf2')

#plt.plot(future_predictions[:,0],future_predictions[:,1],'y',label = 'future_predictions')

plt.plot(future_predictions2[:,0],future_predictions2[:,1],'y',label = 'future_predictions2')

plt.title(' future trajectory of 1s ')

plt.legend(['trajectory2','ekf','future2','trajectory2','ekf','ekf2','future1','future2'])

plt.show()

# for position in enumerate(array_1):
#
#     x_data = position[0]
#
#     y_data = position[1]
#






