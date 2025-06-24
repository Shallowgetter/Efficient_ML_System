"""
utils.py
========
Efficient_Computing_System Experiment Platform Utility Collection (v0.1)

Currently provides:
    1. seed_all(seed)         -- Fix random seeds to ensure reproducibility
    2. get_logger(fname, ...) -- Unified log format/level/output endpoints
    3. AverageMeter           -- Metric average value & rolling updates

Functions or classes can be appended to this file later, maintaining modularity and zero CLI calls.
"""

from __future__ import annotations
import os
import warnings
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ------------------------------------------------------------------ #
# 1. Reproduce experiments: Fix random seeds
# ------------------------------------------------------------------ #
def seed_all(seed: int = 1029) -> None:
    """Fix PyTorch / NumPy / Python random seeds to ensure reproducible results."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # multi-GPU environment
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_logger(
    filename: str | Path,
    name: Optional[str] = None,
    level: int | str = "INFO",
    overwrite: bool = True,
    to_stdout: bool = True,
) -> logging.Logger:
    """
    Create and return a logger with a unified format.

    Parameters
    ----------
    filename  : Log file path
    name      : Logger name (None → root)
    level     : 'DEBUG' | 'INFO' | 'WARNING' etc. or corresponding integer
    overwrite : True → rewrite file; False → append
    to_stdout : Whether to synchronize output to terminal
    """
    lvl = logging._nameToLevel[level] if isinstance(level, str) else level
    fmt = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.handlers.clear()  # avoid adding duplicate handlers

    # FileHandler
    log_file = Path(filename)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh_mode = "w" if overwrite else "a"
    fh = logging.FileHandler(log_file, mode=fh_mode, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    # StreamHandler
    if to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(sh)

    return logger



class AverageMeter:
    """Track and update cumulative averages, used for loss/accuracy statistics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        return f"{self.val:.4f} (avg: {self.avg:.4f})"



if __name__ == "__main__":
    # Fix randomness
    seed_all(42)

    # Get logger
    logger = get_logger("logs/example.log", level="DEBUG")
    logger.info("Logger initialized!")

    # Simulate metric updates
    meter = AverageMeter()
    for step in range(5):
        meter.update(val=random.random(), n=1)
        logger.debug(f"Step {step:02d} | {meter}")



def _handle_hidden(n_hidden):
    "Haddle hidden layers from yaml config."
    if type(n_hidden) == int:
        n_layers = 1
        hidden_dim = n_hidden
    elif type(n_hidden) == str:
        n_hidden = n_hidden.split(",")
        n_hidden = [int(x) for x in n_hidden]
        n_layers = len(n_hidden)
        hidden_dim = n_hidden[0]

        if np.std(n_hidden) != 0:
            warnings.warn('use the first hidden num, '
                          'the rest hidden numbers are deprecated', UserWarning)
    else:
        raise TypeError('n_hidden should be a string or a int.')

    return hidden_dim, n_layers



def save_checkpoint(model, optimizer, epoch, loss, accuracy=None, filename='checkpoint.pth', is_best=False, best_filename='model_best.pth'):
    """
    Save a checkpoint of the model and optimizer state.
    
    Parameters
    ----------
    model: torch.nn.Module
        The model to save
    optimizer: torch.optim.Optimizer
        The optimizer
    epoch: int
        Current epoch
    loss: float
        Current loss value
    accuracy: float, optional
        Current accuracy (if available)
    filename: str
        The filename to save the checkpoint
    is_best: bool
        Whether the current model is the best model
    best_filename: str
        The filename to save the best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy

    # Save the current checkpoint
    torch.save(checkpoint, filename)

    # If it is the best model, copy a version

    if is_best:
        import shutil
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    load a checkpoint from a file and load the model and optimizer states.
    
    Parameters
    ----------
    filename: str
        path to the checkpoint file
    model: torch.nn.Module
        model to load parameters into
    optimizer: torch.optim.Optimizer, optional
        optimizer to load state into

    Returns
    -------
    dict
        all information from the checkpoint
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint



