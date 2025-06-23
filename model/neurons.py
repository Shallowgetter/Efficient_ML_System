'''
This file contains neuron layer definitions for the Efficient Computing System.
Mainly including classical SNN neurons:
from __future__ import annotations
    IF (Integrate-and-Fire)
    LIF (Leaky Integrate-and-Fire)
    ALIF (Adaptive Leaky Integrate-and-Fire)
    LTC (Liquid Time Constant)

Usage:
    Unified function: build_neuron(name: str, **kwargs) -> BaseNode
    Example: node = build_neuron("lif", tau_m=20.0, surrogate_alpha=4.0)
    This will return an instance of LIFNode with specified parameters.
'''

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "SurrogateSpike",
    "BaseNode",
    "IFNode",
    "LIFNode",
    "ALIFNode",
    "LTCNode",
]


# -----------------------------------------------------------------------------
#   Surrogate gradient implementation
# -----------------------------------------------------------------------------
class SurrogateSpike(torch.autograd.Function):
    """Heaviside with SIGMOID derivative (α·σ(x)(1-sigma(x)))."""

    @staticmethod
    def forward(ctx, inp: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(inp)
        ctx.alpha = alpha  # type: ignore[attr-defined]
        return (inp > 0.0).to(inp)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, None]:  # type: ignore[override]
        (inp,) = ctx.saved_tensors
        alpha: float = ctx.alpha  # type: ignore[attr-defined]
        sig: Tensor = torch.sigmoid(alpha * inp)
        grad_in = grad_out * alpha * sig * (1.0 - sig)
        return grad_in, None


# -----------------------------------------------------------------------------
#   Base spiking neuron class
# -----------------------------------------------------------------------------
class BaseNode(nn.Module):
    """Common utility base for all spiking neurons."""

    def __init__(
        self,
        threshold: float = 1.0,
        reset: float = 0.0,
        surrogate_alpha: float = 4.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mem", torch.tensor(0.0))
        self.threshold = threshold
        self.reset = reset
        self.surrogate_alpha = surrogate_alpha

    # ------------------------------------------------------------------ utils
    def _init_state(self, x: Tensor) -> None:
        if self.mem.numel() != x.numel():
            self.mem = torch.zeros_like(x)

    def reset_state(self) -> None:
        self.mem.zero_()

    # ------------------------------------------------------------------ core
    def neuron_update(self, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        """Override in subclasses; returns (out_spike, new_mem)."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Supports both single-step and multi-step tensors."""
        if x.dim() == self.mem.dim() + 1:  # treat dim0 as *time*
            outputs = []
            for t in range(x.size(0)):
                out, _ = self.neuron_update(x[t])
                outputs.append(out)
            return torch.stack(outputs, 0)
        else:
            out, _ = self.neuron_update(x)
            return out


# -----------------------------------------------------------------------------
#   Concrete neuron models
# -----------------------------------------------------------------------------
class IFNode(BaseNode):
    """Integrate-and-Fire (no leak)."""

    def neuron_update(self, x_t: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D401
        self._init_state(x_t)
        self.mem = self.mem + x_t
        spike = SurrogateSpike.apply(self.mem - self.threshold, self.surrogate_alpha)
        self.mem = torch.where(spike.bool(), torch.full_like(self.mem, self.reset), self.mem)
        return spike, self.mem


class LIFNode(BaseNode):
    """Leaky Integrate-and-Fire."""

    def __init__(
        self,
        tau_m: float = 20.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tau_m = tau_m
        self.register_buffer("decay", torch.tensor(math.exp(-1.0 / tau_m)))

    def neuron_update(self, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        self._init_state(x_t)
        self.mem = self.mem * self.decay + x_t
        spike = SurrogateSpike.apply(self.mem - self.threshold, self.surrogate_alpha)
        self.mem = torch.where(spike.bool(), torch.full_like(self.mem, self.reset), self.mem)
        return spike, self.mem


class ALIFNode(LIFNode):
    """Adaptive threshold LIF."""

    def __init__(
        self,
        tau_m: float = 20.0,
        tau_a: float = 200.0,
        beta: float = 1.6,
        **kwargs,
    ) -> None:
        super().__init__(tau_m=tau_m, **kwargs)
        self.tau_a = tau_a
        self.beta = beta
        self.register_buffer("a", torch.tensor(0.0))
        self.register_buffer("decay_a", torch.tensor(math.exp(-1.0 / tau_a)))

    def reset_state(self) -> None:  # override to reset a
        super().reset_state()
        self.a.zero_()

    def neuron_update(self, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        self._init_state(x_t)
        self.mem = self.mem * self.decay + x_t - self.beta * self.a
        spike = SurrogateSpike.apply(self.mem - self.threshold, self.surrogate_alpha)

        # update adaptation current
        self.a = self.a * self.decay_a + spike

        self.mem = torch.where(spike.bool(), torch.full_like(self.mem, self.reset), self.mem)
        return spike, self.mem


class LTCNode(BaseNode):
    """Liquid-Time-Constant neuron with learnable, state-dependent τ."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        threshold: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        # parameterise τ(u, x) = Softplus(Wx + Uh + b) + ε
        self.W_tau = nn.Linear(in_features, hidden_features, bias=False)
        self.U_tau = nn.Linear(hidden_features, hidden_features, bias=True)
        self.eps = 1e-3

    def reset_state(self) -> None:
        super().reset_state()
        self.ht = None  # type: ignore[attr-defined]

    def _init_state(self, x: Tensor) -> None:
        if not hasattr(self, "ht") or self.ht is None or self.ht.shape != x.shape:
            self.ht = torch.zeros_like(x)  # type: ignore[attr-defined]
        if self.mem.numel() != x.numel():
            self.mem = torch.zeros_like(x)

    def neuron_update(self, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        self._init_state(x_t)
        tau = torch.softplus(self.W_tau(x_t) + self.U_tau(self.ht)) + self.eps
        decay = torch.exp(-1.0 / tau)
        self.mem = self.mem * decay + x_t
        spike = SurrogateSpike.apply(self.mem - self.threshold, self.surrogate_alpha)
        self.mem = torch.where(spike.bool(), torch.zeros_like(self.mem), self.mem)
        self.ht = self.mem  # update hidden trace
        return spike, self.mem


# -----------------------------------------------------------------------------
#   Handy wrappers (Conv/Linear with pre‑synaptic integration)
# -----------------------------------------------------------------------------
class SpikingLinear(nn.Module):
    """Linear layer followed by a spiking neuron."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        neuron: str | nn.Module = "lif",
        **neuron_kwargs,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.node = self._build_node(neuron, out_features, **neuron_kwargs)

    @staticmethod
    def _build_node(name_or_module, out_features: int, **kwargs):
        if isinstance(name_or_module, nn.Module):
            return name_or_module
        name = name_or_module.lower()
        if name == "if":
            return IFNode(**kwargs)
        elif name == "lif":
            return LIFNode(**kwargs)
        elif name == "alif":
            return ALIFNode(**kwargs)
        elif name == "ltc":
            return LTCNode(in_features=out_features, hidden_features=out_features, **kwargs)
        else:
            raise ValueError(f"Unsupported neuron type: {name_or_module}")

    def reset_state(self) -> None:
        if hasattr(self.node, "reset_state"):
            self.node.reset_state()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.node(self.fc(x))


# -----------------------------------------------------------------------------
#   Exposed factory for simple use‑cases
# -----------------------------------------------------------------------------
NODE_FACTORY = {
    "if": IFNode,
    "lif": LIFNode,
    "alif": ALIFNode,
    "ltc": LTCNode,
}


def build_neuron(name: str, **kwargs) -> BaseNode:  # type: ignore[return-value]
    """Convenience helper mirroring :pydata:`NODE_FACTORY`."""
    name = name.lower()
    if name not in NODE_FACTORY:
        raise KeyError(f"Neuron type '{name}' not found.")
    print(f'Building neuron: {name} with parameters: {kwargs}')
    return NODE_FACTORY[name](**kwargs)


# -----------------------------------------------------------------------------
#   Quick self-test (disabled when imported as module)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(10, 3)  # 10 time-steps, batch=1, features=3 (toy)
    node = LIFNode(tau_m=10.0, surrogate_alpha=2.0)
    out = node(x)
    print(out.sum())