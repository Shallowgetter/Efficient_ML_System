"""
rnn_base.py
-------------

A flexible RNN/GRU/LSTM implementation with support for multi-layer with different hidden sizes.

YAML example:
~~~~~~~~~
```yaml
# conf/model.yaml
model:
  input_size: 128        # Feature dimension at each step
  hidden_size: 256       # Single hidden size for all layers
  # OR
  hidden_size: "256,512" # Different hidden sizes per layer
  num_layers: 2          # Number of stacked layers
  rnn_type: GRU          # {RNN, GRU, LSTM}
  bidirectional: false
  dropout: 0.1
  batch_first: true
  return_sequence: false # Only return features from the last time step
  num_classes: 14        # SHL has 14 types of activities
```

then:
```python
from rnn_base import RNNBase
model = RNNBase.from_config('conf/model.yaml').to(device)
```
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

__all__ = ["RNNBase"]

RNN_TYPE_MAP: Dict[str, type[nn.Module]] = {
    "RNN": nn.RNN,
    "GRU": nn.GRU,
    "LSTM": nn.LSTM,
}


class MultiLayerRNN(nn.Module):
    """Custom multi-layer RNN with support for different hidden sizes per layer."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        rnn_type: str,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        self.num_layers = len(hidden_sizes)
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.hidden_sizes = hidden_sizes
        self.rnn_type = rnn_type
        
        # Create a list of RNN layers
        rnn_cls = RNN_TYPE_MAP[rnn_type]
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layer_input_size = input_size if i == 0 else hidden_sizes[i-1] * (2 if bidirectional else 1)
            layer = rnn_cls(
                input_size=layer_input_size,
                hidden_size=hidden_sizes[i],
                num_layers=1,  # Each module is a single layer
                batch_first=batch_first,
                dropout=0.0,  # No internal dropout, we'll add it between layers
                bidirectional=bidirectional,
            )
            self.layers.append(layer)
            
        # Add dropout between layers if specified
        if dropout > 0 and self.num_layers > 1:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.num_layers - 1)])
        else:
            self.dropouts = None
    
    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        batch_dim = 0 if not self.batch_first else 0
        
        # Parse initial hidden states for each layer
        if h0 is None:
            h0_list = [None] * self.num_layers
        elif isinstance(h0, tuple):  # LSTM case (hidden, cell)
            h, c = h0
            h0_list = [(h[i:i+1], c[i:i+1]) for i in range(self.num_layers)]
        else:  # RNN/GRU case
            h0_list = [h0[i:i+1] for i in range(self.num_layers)]
        
        # Output hidden states from all layers
        all_hn = []
        
        # Pass through each layer
        current_x = x
        for i, layer in enumerate(self.layers):
            output, hn = layer(current_x, h0_list[i])
            
            # Store the hidden state
            all_hn.append(hn)
            
            # Apply dropout if not the last layer
            if self.dropouts is not None and i < self.num_layers - 1:
                current_x = self.dropouts[i](output)
            else:
                current_x = output
        
        # Combine hidden states from all layers
        if self.rnn_type == "LSTM":
            h_list, c_list = zip(*all_hn)
            final_h = torch.cat(h_list, dim=0)
            final_c = torch.cat(c_list, dim=0)
            final_hn = (final_h, final_c)
        else:
            final_hn = torch.cat(all_hn, dim=0)
        
        return current_x, final_hn


class RNNBase(nn.Module):
    """RNN/GRU/LSTM backbone with support for different hidden sizes per layer"""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Union[str, pathlib.Path, Mapping[str, Any]]) -> "RNNBase":
        """Construct model from YAML file or mapping object.

        Parameters
        ----------
        cfg : str | Path | Mapping
            * If path/string → read YAML; file should include top-level key `model`.
            * If mapping → treated directly as model kwargs.
        """
        if isinstance(cfg, (str, pathlib.Path)):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
            cfg_dict = cfg_dict.get("model", cfg_dict)  # Compatible with top-level parameters
        else:
            cfg_dict = dict(cfg)
        return cls(**cfg_dict)

    # ------------------------------------------------------------------
    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, str, List[int]],
        num_layers: int = 1,
        *,
        rnn_type: str = "GRU",
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        return_sequence: bool = False,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.upper() if rnn_type else "GRU"
        if rnn_type not in RNN_TYPE_MAP:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Choose from {list(RNN_TYPE_MAP)}.")

        self.return_sequence = return_sequence
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # Parse hidden sizes
        if isinstance(hidden_size, int):
            # Single value for all layers
            self.hidden_sizes = [hidden_size] * num_layers
        elif isinstance(hidden_size, str):
            # Parse comma-separated values (e.g., "32,64")
            self.hidden_sizes = [int(h.strip()) for h in hidden_size.split(",")]
            if len(self.hidden_sizes) == 1 and num_layers > 1:
                # If only one value provided but multiple layers required
                self.hidden_sizes = self.hidden_sizes * num_layers
            elif len(self.hidden_sizes) < num_layers:
                raise ValueError(f"Specified {len(self.hidden_sizes)} hidden sizes but {num_layers} layers required")
            elif len(self.hidden_sizes) > num_layers:
                # Use only the first num_layers values
                self.hidden_sizes = self.hidden_sizes[:num_layers]
        else:
            # Assume list/tuple of integers
            self.hidden_sizes = list(hidden_size)
            if len(self.hidden_sizes) < num_layers:
                raise ValueError(f"Specified {len(self.hidden_sizes)} hidden sizes but {num_layers} layers required")
            elif len(self.hidden_sizes) > num_layers:
                # Use only the first num_layers values
                self.hidden_sizes = self.hidden_sizes[:num_layers]
        
        self.num_layers = len(self.hidden_sizes)
        
        # Create either standard PyTorch RNN (all layers same size) or custom multi-layer RNN
        if all(h == self.hidden_sizes[0] for h in self.hidden_sizes):
            # All hidden sizes are the same, use standard PyTorch RNN
            rnn_cls = RNN_TYPE_MAP[rnn_type]
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=self.hidden_sizes[0],
                num_layers=self.num_layers,
                batch_first=batch_first,
                dropout=dropout if self.num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            # Different hidden sizes per layer, use custom implementation
            self.rnn = MultiLayerRNN(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=batch_first,
            )

        # Final hidden size is from the last layer
        self.feature_dim = self.hidden_sizes[-1] * (2 if bidirectional else 1)
        self.head: nn.Module = (
            nn.Linear(self.feature_dim, num_classes) if num_classes is not None else nn.Identity()
        )

        self._init_parameters()

    # ------------------------------------------------------------------
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Return logits (or feats) + hidden."""
        y, hn = self.rnn(x, h0)
        feat = y if self.return_sequence else (y[:, -1] if self.batch_first else y[-1])
        out = self.head(feat)
        return out, hn

    # ------------------------------------------------------------------
    def init_hidden(
        self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Zero‑initialised hidden (and cell) state."""
        num_directions = 2 if self.bidirectional else 1
        
        if isinstance(self.rnn, MultiLayerRNN):
            # For custom multi-layer RNN with different hidden sizes
            if self.rnn.rnn_type == "LSTM":
                h_parts = []
                c_parts = []
                for layer_idx, hidden_size in enumerate(self.hidden_sizes):
                    shape = (1 * num_directions, batch_size, hidden_size)
                    h_parts.append(torch.zeros(shape, device=device, dtype=dtype))
                    c_parts.append(torch.zeros(shape, device=device, dtype=dtype))
                return torch.cat(h_parts, dim=0), torch.cat(c_parts, dim=0)
            else:
                h_parts = []
                for layer_idx, hidden_size in enumerate(self.hidden_sizes):
                    shape = (1 * num_directions, batch_size, hidden_size)
                    h_parts.append(torch.zeros(shape, device=device, dtype=dtype))
                return torch.cat(h_parts, dim=0)
        else:
            # For standard PyTorch RNN with same hidden size for all layers
            shape = (self.num_layers * num_directions, batch_size, self.hidden_sizes[0])
            if isinstance(self.rnn, nn.LSTM):
                h0 = torch.zeros(shape, device=device, dtype=dtype)
                c0 = torch.zeros(shape, device=device, dtype=dtype)
                return h0, c0
            else:
                return torch.zeros(shape, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Optional: checkpoint helpers (kept for convenience)
    # ------------------------------------------------------------------
    def save_ckpt(self, path: Union[str, pathlib.Path], step: int, **extra) -> None:
        ckpt = {"state_dict": self.state_dict(), "step": step, "meta": extra}
        torch.save(ckpt, str(path))

    def load_ckpt(self, path: Union[str, pathlib.Path], strict: bool = True) -> dict:
        ckpt = torch.load(str(path), map_location="cpu")
        self.load_state_dict(ckpt["state_dict"], strict=strict)
        return {k: v for k, v in ckpt.items() if k != "state_dict"}


class GRU(RNNBase):
    """GRU model with default parameters."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type="GRU", **kwargs)


class LSTM(RNNBase):
    """LSTM model with default parameters."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type="LSTM", **kwargs)