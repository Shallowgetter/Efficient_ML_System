"""dataset.py

Used for loading and preprocessing the SHL Magnetometer dataset.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class SHLConfig:
    root: Path                       
    users: Optional[Sequence[str]] = None   
    versions: Tuple[str, ...] = (
        "SHLDataset_preview_v1_1",
        "SHLDataset_preview_v1_2",
        "SHLDataset_preview_v1_3",
    )
    positions: Tuple[str, ...] = ("Hand", "Bag", "Hips", "Torso")
    window_size: int = 500          # 5 s * 100 Hz
    step_size: int = 500            
    label_type: str = "coarse"      # "coarse" or "fine"
    drop_null: bool = True          
    dtype: torch.dtype = torch.float32




_MAG_COLS = [7, 8, 9]               # 0-index, corresponding to columns 8-10 [turn0file2]

def parse_motion_file(path: Path) -> np.ndarray:
    """Read a single *_Motion.txt file and return a (N, 3) magnetometer matrix."""
    data = np.loadtxt(path, dtype=np.float32)
    return data[:, _MAG_COLS]       # Only keep MagX, MagY, MagZ


def parse_label_file(path: Path, label_type: str = "coarse") -> np.ndarray:
    """
    Read Label.txt and return a (N,) integer vector.
    label_type = "coarse"  → column 2
               = "fine"    → column 3
    """
    col = 1 if label_type == "coarse" else 2
    labels = np.loadtxt(path, usecols=[col], dtype=np.int16)
    return labels


def load_recording(date_dir: Path,
                   positions: Sequence[str],
                   label_type: str = "coarse") -> Tuple[np.ndarray, np.ndarray]:
    """
    Read magnetometer data from four positions + labels from the same date directory.

    Returns
    -------
    data  : (N, 12) float32
    label : (N,)   int16
    """
    # Concatenate according to positions order
    mag_list: List[np.ndarray] = []
    for pos in positions:
        p = date_dir / f"{pos}_Motion.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing motion file: {p}")
        mag_list.append(parse_motion_file(p))

    # Check length consistency
    lengths = [m.shape[0] for m in mag_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch in {date_dir}: {lengths}")
    data = np.concatenate(mag_list, axis=1)  # (N, 12)

    # Labels
    label_path = date_dir / "Label.txt"
    labels = parse_label_file(label_path, label_type=label_type)
    if labels.shape[0] != data.shape[0]:
        raise ValueError(f"Label length mismatch in {date_dir}")

    return data, labels


def slice_index(total_len: int, window: int, step: int) -> List[int]:
    """Return all window start indices (inclusive, closed interval)."""
    return list(range(0, total_len - window + 1, step))




class SHLMagDataset(Dataset):
    """
    Lazy-loading SHL Magnetometer dataset, slicing only when __getitem__ is called.

    Each sample:
        X : (window_size, 12) torch.float32
        y : int  (majority label in the window)
    """

    def __init__(self,
                 cfg: SHLConfig,
                 transform=None):
        self.cfg = cfg
        self.transform = transform

        self.chunks: List[Dict] = []        # Each recording's {"data":ndarray, "label":ndarray}
        self.index_map: List[Tuple[int, int]] = []  # (chunk_idx, start_idx)

        self._build_index()


    def _iter_date_dirs(self) -> List[Path]:
        """Recursively scan all date directories containing *_Motion.txt files."""
        import glob
        dirs = set()
        for ver in self.cfg.versions:
            for user in (self.cfg.users if self.cfg.users else ["*"]):
                pattern = str(self.cfg.root / ver / user / "*" / "*_Motion.txt")
                for p in glob.glob(pattern):
                    dirs.add(Path(p).parent)
        return sorted(dirs)

    def _build_index(self):
        """Read all recordings and generate indices."""
        date_dirs = self._iter_date_dirs()
        for date_dir in tqdm(date_dirs, desc="Loading data directories"):
            try:
                data, label = load_recording(
                    date_dir,
                    self.cfg.positions,
                    label_type=self.cfg.label_type
                )
            except (FileNotFoundError, ValueError):
                # Print warning if needed
                continue

            chunk_idx = len(self.chunks)
            self.chunks.append({"data": data, "label": label})

            # Generate window indices
            window_indices = slice_index(
                    total_len=data.shape[0],
                    window=self.cfg.window_size,
                    step=self.cfg.step_size)
            
            for start in tqdm(window_indices, desc=f"Processing windows for {date_dir.name}", leave=False):
                # Null label filtering
                if self.cfg.drop_null:
                    window_labels = label[start:start + self.cfg.window_size]
                    if np.all(window_labels == 0):
                        continue
                self.index_map.append((chunk_idx, start))


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        chunk_idx, start = self.index_map[idx]
        chunk = self.chunks[chunk_idx]
        x_np = chunk["data"][start:start + self.cfg.window_size]         # (W, 12)
        y_np = chunk["label"][start:start + self.cfg.window_size]

        # Multiple labels → take mode; if Null exists, keep non-Null mode, otherwise 0
        if self.cfg.drop_null:
            y_non_zero = y_np[y_np != 0]
            y_scalar = int(np.bincount(y_non_zero).argmax()) if y_non_zero.size else 0
        else:
            y_scalar = int(np.bincount(y_np).argmax())

        x = torch.as_tensor(x_np, dtype=self.cfg.dtype)
        y = torch.as_tensor(y_scalar, dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y




def get_dataloader(cfg: SHLConfig,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   transform=None) -> DataLoader:
    """
    Convenience function to construct DataLoader.

    Parameters
    ----------
    cfg         : SHLConfig
    batch_size  : Batch size
    shuffle     : Whether to shuffle (only for training)
    num_workers : Number of PyTorch DataLoader workers
    transform   : Optional data augmentation function
    """
    dataset = SHLMagDataset(cfg, transform=transform)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)



if __name__ == "__main__":

    cfg = SHLConfig(root=Path("/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/SHL"))
    loader = get_dataloader(cfg, batch_size=32)

    x, y = next(iter(loader))
    print("Batch X shape:", x.shape)   # (32, 500, 12)
    print("Batch y shape:", y.shape)
    print(len(loader))  # Number of batches
    print("Label: ", y.unique())  # Unique labels in the batch
    print("First sample X:", x[0])     # (500, 12)
    print("First sample y:", y[0])     # Scalar label
