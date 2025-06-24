"""
According to paper 'Combining Residual and LSTM Recurrent Networks for Transportation Mode Detection Using Multimodal Sensors Integrated in Smartphones', 
the MLP structure is FC(128) - FC(256) - FC(512) - FC(1024) - Softmax(8).
"""

import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2, nclass: int = 8):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, nclass)   # Softmax is applied in the loss function (crossEntropy), not here
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)                  
