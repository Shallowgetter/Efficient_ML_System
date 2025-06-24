"""
According to paper 'Combining Residual and LSTM Recurrent Networks for Transportation Mode Detection Using Multimodal Sensors Integrated in Smartphones', 
the MLP structure is FC(128) - FC(256) - FC(512) - FC(1024) - Softmax(8).
"""

import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.5):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 512)
        self.fc4 = torch.nn.Linear(512, 1024)
        self.fc5 = torch.nn.Linear(1024, 8)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return self.softmax(x)