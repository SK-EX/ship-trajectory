import ast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

# 数据加载和预处理
trajectories = []
file = open("result.txt",'r')
for line in file:
    line = line.strip()
    trajectories.append(line)
tuple_1 = trajectories[0]
tuple_2 = trajectories[2]
data = ast.literal_eval(tuple_1)
data_2 = ast.literal_eval(tuple_2)
# 转换为 numpy 数组
array_1 = np.array(data)
array_2 = np.array(data_2)
x_data = array_2[:, 0]
y_data = array_2[:, 1]
max = np.max(y_data)
min = np.min(y_data)
x_data = (x_data - min) / (max - min)
y_data = (y_data - min) / (max - min)

# 划分训练集和测试集
split_ratio = 0.8
split_idx = int(len(x_data) * split_ratio)
train_data = x_data[:split_idx]
test_data = x_data[split_idx:]


# 序列生成函数
def create_sequences(origin_data, seq_length):
    xs, ys = [], []
    for i in range(len(origin_data) - seq_length):
        xs.append(origin_data[i:i + seq_length])
        ys.append(origin_data[i + seq_length])
    return np.array(xs), np.array(ys)


seq_length = 10
train_x, train_y = create_sequences(train_data, seq_length)
test_x, test_y = create_sequences(test_data, seq_length)

# 转换为PyTorch张量
train_x = torch.tensor(train_x[:, :, None], dtype=torch.float32)  # 添加特征维度
train_y = torch.tensor(train_y[:, None], dtype=torch.float32)
test_x = torch.tensor(test_x[:, :, None], dtype=torch.float32)
test_y = torch.tensor(test_y[:, None], dtype=torch.float32)

# 创建DataLoader
batch_size = 32
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 自动初始化隐藏状态
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步的输出
        return self.fc(out)


# 模型初始化
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 60

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    # 训练集预测
    train_pred = model(train_x)
    train_loss = criterion(train_pred, train_y)

    # 测试集预测
    test_pred = model(test_x)
    test_loss = criterion(test_pred, test_y)

    print(f'Final Training Loss: {train_loss.item():.4f}')
    print(f'Test Loss: {test_loss.item():.4f}')


# 反归一化函数
def inverse_normalize(normalized, max_val, min_val):
    return normalized * (max_val - min_val) + min_val


# 可视化结果
plt.figure(figsize=(12, 6))

# 处理训练集结果
train_time = np.arange(seq_length, split_idx)
pred_train = inverse_normalize(train_pred.squeeze().numpy(), max, min)
true_train = inverse_normalize(train_data[seq_length:], max, min)

# 处理测试集结果
test_time = np.arange(split_idx + seq_length, len(x_data))
pred_test = inverse_normalize(test_pred.squeeze().numpy(), max, min)
true_test = inverse_normalize(test_data[seq_length:], max, min)

# 绘制结果
plt.plot(time[seq_length:split_idx], true_train, label='Original Train')
plt.plot(time[split_idx + seq_length:], true_test, label='Original Test')
plt.plot(time[seq_length:split_idx], pred_train, '--', label='Predicted Train')
plt.plot(time[split_idx + seq_length:], pred_test, '--', label='Predicted Test')

plt.title('Original vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()