import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM ,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x,h0,c0):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (hn,cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, : ]
        out = self.fc(out)

        return out,hn,cn


def load_and_preprocess_data(file_path):
    dataset = np.loadtxt(file_path, delimiter=',')
    timestamps = dataset[:, 0]
    feature1 = np.array(dataset[:, 1])
    feature2 = np.array(dataset[:, 2])
    feature3 = np.array(dataset[:, 3])
    return timestamps, feature1, feature2, feature3


def normalize_data(input_array):
    data_min = np.min(input_array)
    data_max = np.max(input_array)
    return (input_array - data_min) / (data_max - data_min)


def generate_sequences(input_sequence, window_size):
    sequence_list = []
    target_list = []
    for idx in range(len(input_sequence) - window_size):
        seq_window = input_sequence[idx:(idx + window_size)]
        target_value = input_sequence[idx + window_size]
        sequence_list.append(seq_window)
        target_list.append(target_value)
    return np.array(sequence_list), np.array(target_list)


def train_model(model, criterion, optimizer, input_tensor, target_tensor, num_epochs):
    hidden_state, cell_state = None, None
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        predictions, hidden_state, cell_state = model(input_tensor, hidden_state, cell_state)

        loss = criterion(predictions, target_tensor)
        loss.backward()
        optimizer.step()

        hidden_state = hidden_state.detach()
        cell_state = cell_state.detach()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return hidden_state, cell_state


def visualize_results(ground_truth, predictions, start_idx):
    time_points = np.arange(start_idx, len(ground_truth))
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, ground_truth[start_idx:], label='Original Data')
    plt.plot(time_points, predictions.detach().numpy(), label='Model Predictions')
    plt.title('Comparison of Original and Predicted Values')
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()


# Main execution
timestamps, f1_data, f2_data, f3_data = load_and_preprocess_data('M3_O1.6h56H6T1.9r2.csv')
normalized_data = normalize_data(f1_data)

sequence_window = 10
X, y = generate_sequences(normalized_data, sequence_window)
X_tensor = torch.tensor(X[:, :, None], dtype=torch.float32)
y_tensor = torch.tensor(y[:, None], dtype=torch.float32)

model_config = {
    'input_dim': 1,
    'hidden_dim': 100,
    'layer_count': 1,
    'output_dim': 1
}

prediction_model = LSTM(model_config['input_dim'],
                        model_config['hidden_dim'],
                        model_config['layer_count'],
                        model_config['output_dim'])

loss_function = nn.MSELoss()
model_optimizer = torch.optim.Adam(prediction_model.parameters(), lr=0.01)
training_epochs = 60

final_hidden, final_cell = train_model(prediction_model,
                                       loss_function,
                                       model_optimizer,
                                       X_tensor,
                                       y_tensor,
                                       training_epochs)

prediction_model.eval()
model_predictions, _, _ = prediction_model(X_tensor, final_hidden, final_cell)
visualize_results(normalized_data, model_predictions, sequence_window)


