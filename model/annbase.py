"""
ANN-based model using PyTorch.

Mainly including:
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)
- TCN (Temporal Convolutional Network)

Using yaml to save network configurations.
# yaml ex.
#model:
#  name: GRUNet
#  hidden_size: 64
#  num_layers: 2
#  dropout: 0.5
#  bidirectional: true
#activation: ReLU
"""

from __future__ import annotations
from utils.utils import _handle_hidden

import torch
import numpy as np
import torch.nn.modules.activation


__all__ = [
    "GRUNet",
    "LSTMNet",
    "TCNNet",
]

def get_network(network_name):

    if network_name in __all__:
        return globals()[network_name]
    else:
        raise ValueError(f"Network '{network_name}' is not defined in {__name__}. Available networks: {', '.join(__all__)}.")
    


class GRUNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 dropout=None, activation=None):
        super(GRUNet, self).__init__()

        hidden_size, num_layers = _handle_hidden(hidden_size)

        if dropout is None:
            dropout = 0.0

        self.gru = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=True)
        
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h = self.gru(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)

        return out



class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 dropout=None, activation=None):
        super(LSTMNet, self).__init__()

        hidden_size, num_layers = _handle_hidden(hidden_size)

        if dropout is None:
            dropout = 0.0

        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  batch_first=True,
                                  bidirectional=True)
        
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)

        return out
    

    
class TCNNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 kernel_size=3, dropout=None, activation='ReLU'):
        super(TCNNet, self).__init__()

        hidden_size, num_layers = _handle_hidden(hidden_size)

        if dropout is None:
            dropout = 0.0

        self.tcn = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            torch.nn.ReLU() if activation == 'ReLU' else torch.nn.Identity(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_size, output_size, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)  # Change to (batch_size, input_size, seq_length)
        out = self.tcn(x)
        out = out.transpose(1, 2)  # Change back to (batch_size, seq_length, output_size)

        return out[:, -1, :]  # Take the last time step