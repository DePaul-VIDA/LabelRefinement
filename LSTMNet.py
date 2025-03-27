import torch.nn as nn

class PointsLSTMNet(nn.Module):
    def __init__(self, count_people):
        super().__init__()
        self.lstm = nn.LSTM(input_size=(9 * count_people), hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
