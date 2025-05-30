import  numpy as np
import ast

import torch
from torch import float32
from torch.nn import MSELoss

from LSTM import LSTMModel
#处理数据
# 数据加载和预处理
def load_data(trajectory_str):
    data = list(ast.literal_eval(trajectory_str))
    array = np.array(data, dtype=np.float32)
    x_data = array[:, 0].reshape(-1, 1)
    y_data = array[:, 1].reshape(-1, 1)
    return x_data, y_data

# 加载数据
with open("result.txt", 'r') as file:
    trajectories = [line.strip() for line in file]

x_data, y_data = load_data(trajectories[0])  # 使用第一条轨迹

print(np.size(x_data))
train_x_data = x_data[0:int(0.8 * np.size(x_data)) ,: ]
train_y_data = y_data[0:int(0.8 * np.size(y_data)), : ]

test_x_data =  x_data[int(0.8 * np.size(x_data)):, :]
test_y_data = y_data[int(0.8 * np.size(y_data)):, :]

max_epochs = 70
input_dim = 1
hidden_size = 60
layer_dim = 2
output_size = 1

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_dim,hidden_size, layer_dim, output_size).to(device)

tensor_x = torch.tensor(train_x_data, dtype=float32).unsqueeze(1).to(device)
tensor_y = torch.tensor(train_y_data, dtype=float32).unsqueeze(1).to(device)

tensor_x_test = torch.tensor(test_x_data, dtype=float32).unsqueeze(1).to(device)
tensor_y_test = torch.tensor(test_y_data, dtype=float32).unsqueeze(1).to(device)

print(tensor_x_test.shape
      ,
      tensor_y_test.shape)
c0 , h0= None, None

print(model)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = MSELoss()
for epoch in range(max_epochs):
    optimizer.step()
    optimizer.zero_grad()
    outputs, hn, cn = model(tensor_x,(h0,c0))
    loss = criterion(outputs, tensor_y)
    loss.backward()
    print(f'epoch{epoch}/{max_epochs}, train_loss = {loss}')

