"""
According to paper 'Combining Residual and LSTM Recurrent Networks for Transportation Mode Detection Using Multimodal Sensors Integrated in Smartphones', 
the CNN structure is Each Element [C(64)-P(2)-C(128)-P(2)]-C(32)-P(2)-MLP-Softmax.
"""

import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Conv(3) → BN → ReLU → Conv(3) → BN
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels, momentum=0.3)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels, momentum=0.3)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels, momentum=0.3)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity          
        return F.relu(out)


class _elementWiseBrach(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


class _ElementBranchResidual(nn.Sequential):
    """
    ResidualBlock + MaxPool
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__(
            ResidualBlock(in_channels,  out_channels, stride=1),
            ResidualBlock(out_channels, out_channels, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


class CNN(nn.Module):

    def __init__(
            self,
            num_elements: int = 8,
            window_size: int = 500,
            num_classes: int = 8,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.num_elements = num_elements
        self.window_size = window_size

        # Each element branch
        self.element_branches = nn.ModuleList([
            _elementWiseBrach(1, 64) for _ in range(num_elements)
        ])

        # Global branch
        self.global_branch = nn.Sequential(
            nn.Conv1d(num_elements * 128, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(32 * (window_size // 8), 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # x shape ->: (batch_size, num_elements, window_size)
        batch_size = x.size(0)

        feats: List[torch.Tensor] = []
        for i, branch in enumerate(self.element_branches):
            element_x = x[:, i:i + 1, :]
            # print("element_x shape:", element_x.shape)  # Debugging line
            feats.append(branch(element_x))

        feats = torch.cat(feats, dim=1)  # Concatenate along channel dimension
        feats = self.global_branch(feats)
        feats = feats.view(batch_size, -1)  # Flatten for fully connected layer
        logits = self.fc(feats)
        return logits
    

class CNNGAP(nn.Module):
    """
    CNN with Global Average Pooling and Residual Blocks.
    """
    def __init__(self,
                 num_elements: int = 20,
                 window_size: int = 300,
                 num_classes: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.branches = nn.ModuleList([
            _ElementBranchResidual(in_channels=1, out_channels=32)
            for _ in range(num_elements)
        ])

        self.global_branch = nn.Sequential(
            nn.Conv1d(32 * num_elements, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64, momentum=0.3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)      # (B,64,1)
        self.fc  = nn.Linear(64, num_classes)   # (B,8)

    def forward(self, x):                       # x:(B,T,C)
        B, T, C = x.shape
        assert C == len(self.branches), "Channels of input must match number of branches."

        feats = []
        for i, branch in enumerate(self.branches):
            element_x = x[:, :, i].unsqueeze(1)
            feats.append(branch(element_x))

        feats = torch.cat(feats, dim=1)
        feats = self.global_branch(feats)       # (B,64,L')
        feats = self.gap(feats).squeeze(-1)     # (B,64)
        return self.fc(feats)                   # (B,8)



# if __name__ == "__main__":
#     B, T, C = 8, 450, 10
#     model = CNN(num_elements=C, window_size=T, num_classes=8)
#     x = torch.randn(B, T, C)
#     y = model(x)
#     print("logits:", y.shape) # (B, num_classes)
