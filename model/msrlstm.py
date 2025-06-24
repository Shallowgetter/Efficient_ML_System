import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResidualBlock1D(nn.Module):
    """Simple 1‑D residual block used in the original TensorFlow MSRLSTM.

    The block applies two Conv1D‑>BN‑>ReLU layers with a residual connection.
    Padding is set to preserve the temporal length ("same" convolution).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size // 2) * dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size // 2) * dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        # projection if channels mismatch
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):  # x: [B, C, T]
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.act(out + identity)
        return out

class SelfAttention(nn.Module):
    """Luong‑style additive attention over the temporal dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5

    def forward(self, h):  # h: [B, T, H]
        q = self.query(h)  # [B, T, H]
        k = self.key(h)
        v = self.value(h)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, T, T]
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, v)  # [B, T, H]
        return context, attn

class MSRLSTM(nn.Module):
    """PyTorch re‑implementation of the TensorFlow MSRLSTM network.

    The model follows the description in Wang et al. (IEEE T‑ITS 2021):
        * Residual 1‑D CNNs per sensor element
        * A sensor‑fusion convolutional layer
        * Bidirectional LSTM(s)
        * Temporal attention
        * MLP classifier

    Args
    ----
    in_channels: number of raw feature channels (e.g. 23 for SHL).
    base_channels: width of convolutional blocks.
    num_res_blocks: list with number of residual blocks per stage.
    lstm_hidden: hidden units in LSTM.
    lstm_layers: stacked LSTM layers.
    num_classes: number of transportation modes.
    dropout: dropout probability.
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_res_blocks: List[int] = [2, 2],
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 num_classes: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        # --- Local residual feature extractors (per sensor element) ---
        blocks = [ResidualBlock1D(in_channels, base_channels, kernel_size=3, dropout=dropout)]
        for _ in range(num_res_blocks[0] - 1):
            blocks.append(ResidualBlock1D(base_channels, base_channels, kernel_size=3, dropout=dropout))
        self.local_extractor = nn.Sequential(*blocks)

        # --- Global residual fusion ---
        fusion_blocks = [ResidualBlock1D(base_channels, base_channels * 2, kernel_size=3, dropout=dropout)]
        for _ in range(num_res_blocks[1] - 1):
            fusion_blocks.append(ResidualBlock1D(base_channels * 2, base_channels * 2, kernel_size=3, dropout=dropout))
        self.fusion_extractor = nn.Sequential(*fusion_blocks)

        # --- BiLSTM over time ---
        self.lstm = nn.LSTM(input_size=base_channels * 2,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        # --- Attention ---
        self.attention = SelfAttention(lstm_hidden * 2)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B, T, C]
        # Rearrange for Conv1D ➜ [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.local_extractor(x)
        x = self.fusion_extractor(x)          # [B, C_fusion, T]
        # Prepare for LSTM ➜ [B, T, C_fusion]
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)            # [B, T, 2*H]
        ctx, attn = self.attention(lstm_out)  # same shape as lstm_out
        # Temporal average of context vectors
        ctx = ctx.mean(dim=1)                 # [B, 2*H]
        logits = self.classifier(ctx)         # [B, num_classes]
        return logits, attn

if __name__ == "__main__":
    # Quick shape check
    batch, seq_len, feat = 1024, 300, 32
    model = MSRLSTM(in_channels=feat)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model structure:", model)
    dummy = torch.randn(batch, seq_len, feat)
    out, attn = model(dummy)
    print("logits:", out.shape, "attn:", attn.shape)
