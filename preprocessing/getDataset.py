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
    step_size: Optional[int] = None  # If None, calculated from overlap
    overlap: float = 0.0            # overlap for window between 0.0 and 1.0
    label_type: str = "coarse"      # "coarse" or "fine"
    drop_null: bool = True          
    dtype: torch.dtype = torch.float32


_ACC_COLS = [2, 3, 4]               # 0-index, corresponding to columns 3-5 [turn0file2]
_GYRO_COLS = [5, 6, 7]              # 0-index, corresponding to columns 6-8 [turn0file2]
_MAG_COLS = [7, 8, 9]               # 0-index, corresponding to columns 8-10 [turn0file2]


def parse_motion_file(path: Path) -> np.ndarray:
    """Read a single *_Motion.txt file and return a (N, 3) magnetometer matrix."""
    data = np.loadtxt(path, dtype=np.float32)
    return data[:,_ACC_COLS + _GYRO_COLS + _MAG_COLS]       # Keep AccX, AccY, AccZ, GYRO_X, GYRO_Y, GYRO_Z, MagX, MagY, MagZ


def parse_label_file(path: Path, label_type: str = "coarse") -> np.ndarray:
    """
    Read Label.txt and return a (N,) integer vector.
    label_type = "coarse"  → column 2
               = "fine"    → column 3
    """
    col = 1 if label_type == "coarse" else 2
    labels = np.loadtxt(path, usecols=[col], dtype=np.int16)
    return labels

def normalize_features(data: np.ndarray) -> np.ndarray:
    """
    Normalize the features in the data matrix to zero mean and unit variance.
    Assumes data is of shape (N, 12) where columns 0-2 are MagX, MagY, MagZ for each position.
    """
    # Calculate mean and std for each column
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Avoid division by zero
    stds[stds == 0] = 1.0

    # Normalize
    normalized_data = (data - means) / stds
    return normalized_data

def load_recording(date_dir: Path,
                   positions: Sequence[str],
                   label_type: str = "coarse") -> Tuple[np.ndarray, np.ndarray]:
    """
    Read sensor data from four positions + labels from the same date directory.

    Returns
    -------
    data  : (N, 36) float32 - 9 features (3 acc, 3 gyro, 3 mag) × 4 positions
    label : (N,)   int16
    """
    # Concatenate according to positions order
    sensor_list: List[np.ndarray] = []
    for pos in positions:
        p = date_dir / f"{pos}_Motion.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing motion file: {p}")
        sensor_list.append(parse_motion_file(p))

    # Check length consistency
    lengths = [m.shape[0] for m in sensor_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch in {date_dir}: {lengths}")
    data = np.concatenate(sensor_list, axis=1)  # (N, 36) - 9 features × 4 positions

    # Labels
    label_path = date_dir / "Label.txt"
    labels = parse_label_file(label_path, label_type=label_type)
    if labels.shape[0] != data.shape[0]:
        raise ValueError(f"Label length mismatch in {date_dir}")

    return data, labels


def slice_index(total_len: int, window: int, step: Optional[int] = None, overlap: float = 0.0) -> List[int]:
    """
    Return all window start indices (inclusive, closed interval).
    
    Parameters
    ----------
    total_len : Total length of the data
    window : Window size
    step : Step size, if None, calculated from overlap
    overlap : Rate of overlap between windows (0.0 to 1.0)
    
    Returns
    -------
    List[int] : List of window start indices
    """
    if step is not None:
        # If step is provided, use it directly
        return list(range(0, total_len - window + 1, step))
    else:
        # Otherwise, calculate step from overlap
        # step = window * (1 - overlap)
        computed_step = int(window * (1 - overlap))
        # Ensure step is at least 1 to avoid infinite loop
        step = max(1, computed_step)
        return list(range(0, total_len - window + 1, step))


def process_and_save_dataset(cfg: SHLConfig, 
                             output_path: str, 
                             train_ratio: float = 0.8) -> None:
    """
    Process the dataset, split into train/test sets, and save as npz files.
    
    Parameters
    ----------
    cfg: SHLConfig
        Configuration for dataset processing
    output_path: str
        Path where to save the npz files
    train_ratio: float
        Ratio of data to use for training (default 0.8)
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all windows and labels
    all_windows = []
    all_labels = []
    
    date_dirs = _iter_date_dirs(cfg)
    for date_dir in tqdm(date_dirs, desc="Loading data directories"):
        try:
            data, labels = load_recording(
                date_dir,
                cfg.positions,
                label_type=cfg.label_type
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing {date_dir}: {e}")
            continue

        # generate windows index，support overlap
        window_indices = slice_index(
            total_len=data.shape[0],
            window=cfg.window_size,
            step=cfg.step_size,
            overlap=cfg.overlap
        )
        
        # Process each window
        for start in window_indices:
            end = start + cfg.window_size
            if end > data.shape[0]:
                continue
                
            window_data = data[start:end]
            window_labels = labels[start:end]
            
            # Null label filtering - overlook windows with all labels as 0
            if cfg.drop_null and np.all(window_labels == 0):
                continue
                
            # Multiple labels → take mode
            if cfg.drop_null:
                y_non_zero = window_labels[window_labels != 0]
                if y_non_zero.size == 0:  # This case theoretically shouldn't happen, but added for safety
                    continue
                y_scalar = int(np.bincount(y_non_zero).argmax())
                # Additional check to ensure y_scalar is not 0
                if y_scalar == 0:
                    continue
            else:
                y_scalar = int(np.bincount(window_labels).argmax())

            # Store the window and its label
            all_windows.append(window_data)
            all_labels.append(y_scalar)
    
    # Convert to numpy arrays
    all_windows = np.array(all_windows, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int16)
    
    # Reshape from (n_samples, window_size, n_features) to (n_samples, n_features, window_size)
    all_windows = np.transpose(all_windows, (0, 2, 1))
    
    # Randomly shuffle data
    indices = np.random.permutation(len(all_windows))
    all_windows = all_windows[indices]
    all_labels = all_labels[indices]
    
    # Split into train and test
    split_idx = int(len(all_windows) * train_ratio)
    train_x = all_windows[:split_idx]
    train_y = all_labels[:split_idx]
    test_x = all_windows[split_idx:]
    test_y = all_labels[split_idx:]
    
    # Save to npz files
    np.savez(
        output_dir / "shl_train.npz",
        x=train_x,
        y=train_y
    )
    np.savez(
        output_dir / "shl_test.npz",
        x=test_x,
        y=test_y
    )
    
    print(f"Dataset saved. Train shape: {train_x.shape}, Test shape: {test_x.shape}")


def _iter_date_dirs(cfg: SHLConfig) -> List[Path]:
    """Recursively scan all date directories containing *_Motion.txt files."""
    import glob
    dirs = set()
    for ver in cfg.versions:
        for user in (cfg.users if cfg.users else ["*"]):
            pattern = str(cfg.root / ver / user / "*" / "*_Motion.txt")
            for p in glob.glob(pattern):
                dirs.add(Path(p).parent)
    return sorted(dirs)


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

            # Generate window indices with overlap support
            window_indices = slice_index(
                    total_len=data.shape[0],
                    window=self.cfg.window_size,
                    step=self.cfg.step_size,
                    overlap=self.cfg.overlap)
            
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


class SHLNpzDataset(Dataset):
    """
    Dataset that loads preprocessed NPZ files containing SHL sensor data.
    
    Each sample:
        X : (features, window_size) torch.float32
        y : int (class label)
    """
    
    def __init__(self, 
                 npz_path: str,
                 dtype: torch.dtype = torch.float32,
                 transform=None):
        """
        Initialize the dataset from a npz file.
        
        Parameters
        ----------
        npz_path: str
            Path to the npz file
        dtype: torch.dtype
            Data type for tensor conversion
        transform: callable
            Optional transform to be applied on a sample
        """
        data = np.load(npz_path)
        self.x = data['x']  # (n_samples, n_features, window_size)
        self.y = data['y']  # (n_samples,)
        self.dtype = dtype
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=self.dtype)
        y = torch.as_tensor(self.y[idx] - 1, dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def get_npz_dataloader(npz_path: str,
                       batch_size: int = 32,
                       shuffle: bool = True,
                       num_workers: int = 4,
                       dtype: torch.dtype = torch.float32,
                       transform=None) -> DataLoader:
    """
    Create a DataLoader from a preprocessed npz file.
    
    Parameters
    ----------
    npz_path: str
        Path to the npz file
    batch_size: int
        Batch size
    shuffle: bool
        Whether to shuffle the data
    num_workers: int
        Number of worker processes
    dtype: torch.dtype
        Data type for tensor conversion
    transform: callable
        Optional transform to be applied on a sample
    """
    dataset = SHLNpzDataset(
        npz_path=npz_path,
        dtype=dtype,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    # Define the configuration
    cfg = SHLConfig(
        root=Path("/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/SHL"),
        window_size=300,
        overlap=0.3
        )
    
    # process and save the dataset
    process_and_save_dataset(
        cfg=cfg,
        output_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data",
        train_ratio=0.8
    )
    
    # load the dataset using DataLoader
    train_loader = get_npz_dataloader(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_train.npz",
        batch_size=1024,
        shuffle=True
    )

    
    # test_loader = get_npz_dataloader(
    #     npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_test.npz",
    #     batch_size=32,
    #     shuffle=False
    # )
    
    # debug
    x, y = next(iter(train_loader))
    print("Training batch shape:", x.shape)  #  (batch_size, features, window_size)
    print("Training batch y shape:", y.shape)  #  (batch_size,)
    print("Labels:", y.unique())
