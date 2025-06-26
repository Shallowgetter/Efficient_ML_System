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


_ACC_COLS = [1, 2, 3]               # 0-index, corresponding to columns 2-4 [turn0file2]
_GYRO_COLS = [4, 5, 6]              # 0-index, corresponding to columns 5-7 [turn0file2]
_MAG_COLS = [7, 8, 9]               # 0-index, corresponding to columns 8-10 [turn0file2]


def parse_motion_file(path: Path, mag_only: bool = False) -> np.ndarray:
    """Read a single *_Motion.txt file and return a (N, 3) magnetometer matrix."""
    data = np.loadtxt(path, dtype=np.float32)
    if mag_only:
        return data[:, _MAG_COLS]  # Keep only magnetometer columns
    return data[:, _ACC_COLS + _GYRO_COLS + _MAG_COLS]  # Keep all columns


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
                   label_type: str = "coarse",
                   mag_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read sensor data from positions + labels from the same date directory.
    
    Parameters
    ----------
    date_dir : Path to date directory
    positions : Sequence of positions (e.g., ["Hand", "Bag"])
    label_type : Type of labels to load ("coarse" or "fine")
    mag_only : If True, load only magnetometer data
    
    Returns
    -------
    data : numpy array of sensor data
    label : numpy array of labels
    """
    # Concatenate according to positions order
    sensor_list: List[np.ndarray] = []
    for pos in positions:
        p = date_dir / f"{pos}_Motion.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing motion file: {p}")
        sensor_list.append(parse_motion_file(p, mag_only=mag_only))

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


def slice_index(total_len: int, window: int, step: Optional[int] = None, overlap: float = 0.3) -> List[int]:
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


def process_and_save_dataset(cfg: SHLConfig,
                             output_path: str,
                             train_ratio: float = 0.8,
                             seed: int = 42) -> None:
    """
    1) Process raw SHL recordings and save as NPZ files.
    2) Split into training and validation sets based on directory count.
    Parameters
    ----------
    cfg : SHLConfig
        Configuration object containing dataset parameters.
    output_path : str
        Path to save the processed NPZ files.
    train_ratio : float
        Ratio of training set size to total dataset size (default 0.8).
    seed : int
        Random seed for shuffling directories (default 42).
    """
    rng = np.random.RandomState(seed)
    date_dirs = _iter_date_dirs(cfg)
    rng.shuffle(date_dirs)

    # 收集所有目录的信息
    dir_info = []
    all_labels_set = set()
    
    for date_dir in tqdm(date_dirs, desc="Loading recordings"):
        try:
            data, labels = load_recording(date_dir, cfg.positions, cfg.label_type)
            
            if cfg.drop_null:
                valid_indices = labels != 0
                if not np.any(valid_indices):
                    continue
            filtered_data = data[valid_indices] if cfg.drop_null else data
            filtered_labels = labels[valid_indices] if cfg.drop_null else labels

            normalized_data = normalize_features(filtered_data)

            data = normalized_data
            labels = filtered_labels

        except (FileNotFoundError, ValueError):
            continue

        # collect unique labels
        if cfg.drop_null:
            unique_labels = set(labels[labels != 0].tolist())
        else:
            unique_labels = set(labels.tolist())

        if len(unique_labels) == 0:
            continue

        dir_info.append({
            "dir": date_dir,
            "data": data,
            "labels": labels,
            "unique_labels": unique_labels
        })
        
        all_labels_set.update(unique_labels)

    if len(all_labels_set) < 8:
        raise RuntimeError(f"Only contains {len(all_labels_set)} coarse categories, "
                           f"please check label_type and data integrity!")

    # Derive target number of training directories
    target_train_dirs = int(len(dir_info) * train_ratio)
    
    # Balance the number of training and test directories
    sorted_dir_info = sorted(dir_info, key=lambda x: len(x["data"]))
    
    train_bins = sorted_dir_info[:target_train_dirs]
    test_bins = sorted_dir_info[target_train_dirs:]

    # Ensure training set contains all labels
    train_label_set = set().union(*[d["unique_labels"] for d in train_bins])
    missing_labels = all_labels_set - train_label_set
    
    if missing_labels:
        # Move directories containing missing labels from test to train
        for lbl in list(missing_labels):
            for idx, info in enumerate(test_bins):
                if lbl in info["unique_labels"]:
                    train_bins.append(info)
                    test_bins.pop(idx)
                    break

        # Re-check
        train_label_set = set().union(*[d["unique_labels"] for d in train_bins])
        assert all_labels_set - train_label_set == set(), \
            "Still lack of some labels in training set after balancing!"

    def _concat_recordings(bins):
        """Concatenate data and labels from a list of bins."""
        data_list = [b["data"] for b in bins]
        labels_list = [b["labels"] for b in bins]
        
        # Connect all data and labels
        all_data = np.concatenate(data_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        
        return all_data, all_labels

    train_data, train_labels = _concat_recordings(train_bins)
    test_data, test_labels = _concat_recordings(test_bins)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map labels to 0-indexed integers
    # train_labels_mapped = train_labels - 1
    # test_labels_mapped = test_labels - 1

    # Here if -1 -> previous 0 become -1
    train_labels_mapped = train_labels.copy()
    test_labels_mapped = test_labels.copy()

    # Save as NPZ files without windowing 
    np.savez(out_dir / "shl_train.npz", 
             data=train_data.astype(np.float32), 
             labels=train_labels_mapped.astype(np.int16),
             )
    
    np.savez(out_dir / "shl_validation.npz", 
             data=test_data.astype(np.float32), 
             labels=test_labels_mapped.astype(np.int16))
    

    print(f"Data saved to {out_dir}.")
    print(f"Training set: {len(train_data)} time steps")
    print(f"Validation set: {len(test_data)} time steps")
    print(f"Training set directories: {len(train_bins)}")
    print(f"Validation set directories: {len(test_bins)}")
    print(f"Training set label set: {sorted(set(train_labels_mapped))}")




# ----------------------------------------------------------------------------- # 
# -------------------- Dataset for SHL Mag-Only -------------------- #
# ----------------------------------------------------------------------------- # 
class SHLMagDataset(Dataset):
    """
    Dataset that loads SHL data from NPZ files, extracting only magnetometer features.
    
    Each sample:
        X : (window_size, features) torch.float32
        y : int (class label)
    """
    
    def __init__(self, 
                 npz_path: str,
                 window_size: int = 500,
                 overlap: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 transform=None):
        """
        Initialize the dataset from a npz file, focusing on magnetometer data.
        
        Parameters
        ----------
        npz_path: str
            Path to the npz file
        window_size: int
            Size of each window segment
        overlap: float
            Overlap between consecutive windows (0.0 to 1.0)
        dtype: torch.dtype
            Data type for tensor conversion
        transform: callable
            Optional transform to be applied on a sample
        """
        data = np.load(npz_path)
        x_full = data['data']  # (n_samples, n_features)
        y_full = data['labels']  # (n_samples,)
        
        # Extract only magnetometer columns (3 axes × 4 positions = 12 features)
        mag_indices = []
        for pos_idx in range(4):  # 4 positions
            base_idx = pos_idx * 9  # 9 features per position
            mag_start = base_idx + 6  # Magnetometer starts at index 6 (0-indexed)
            mag_indices.extend([mag_start, mag_start + 1, mag_start + 2])
        
        # Extract magnetometer data
        x_mag = x_full[:, mag_indices]
        
        # 计算窗口步长和数量（矢量化分窗）
        step = int(window_size * (1 - overlap))
        n_samples, n_features = x_mag.shape
        n_windows = (n_samples - window_size) // step + 1
        
        if n_windows <= 0:
            raise ValueError(f"Window size {window_size} is too large for data with {n_samples} samples")
        
        # 创建窗口索引
        indices = np.arange(window_size)[None, :] + step * np.arange(n_windows)[:, None]
        
        # 矢量化分窗
        self.x = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
        for i in range(n_features):
            self.x[:, :, i] = x_mag[indices, i]
        
        # 对应窗口的标签
        self.y = y_full[indices[:, 0]]
        
        self.dtype = dtype
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=self.dtype)
        y = torch.as_tensor(self.y[idx], dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


# ----------------------------------------------------------------------------- # 
# -------------------- Dataset for SHL NPZ files -------------------- #
# ----------------------------------------------------------------------------- #
class SHLNpzDataset(Dataset):
    """
    Dataset that loads preprocessed NPZ files containing SHL sensor data.
    
    Each sample:
        X : (window_size, features) torch.float32
        y : int (class label)
    """
    
    def __init__(self, 
                 npz_path: str,
                 window_size: int = 500,
                 overlap: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 transform=None):
        """
        Initialize the dataset from a npz file.
        
        Parameters
        ----------
        npz_path: str
            Path to the npz file
        window_size: int
            Size of each window segment
        overlap: float
            Overlap between consecutive windows (0.0 to 1.0)
        dtype: torch.dtype
            Data type for tensor conversion
        transform: callable
            Optional transform to be applied on a sample
        """
        data = np.load(npz_path)
        x_data = data['data']  # (n_samples, n_features)
        y_data = data['labels']  # (n_samples,)
        
        # 计算窗口步长和数量（矢量化分窗）
        step = int(window_size * (1 - overlap))
        n_samples, n_features = x_data.shape
        n_windows = (n_samples - window_size) // step + 1
        
        if n_windows <= 0:
            raise ValueError(f"Window size {window_size} is too large for data with {n_samples} samples")
        
        # 创建窗口索引
        indices = np.arange(window_size)[None, :] + step * np.arange(n_windows)[:, None]
        
        # 矢量化分窗
        self.x = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
        for i in range(n_features):
            self.x[:, :, i] = x_data[indices, i]
        
        # 对应窗口的标签
        self.y = y_data[indices[:, 0]]
        
        self.dtype = dtype
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=self.dtype)
        y = torch.as_tensor(self.y[idx], dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def get_mag_dataloader(npz_path: str,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4,
                      window_size: int = 500,
                      overlap: float = 0.0,
                      dtype: torch.dtype = torch.float32,
                      transform=None) -> DataLoader:
    """
    Convenience function to construct DataLoader from npz file, focusing on magnetometer data.
    
    Parameters
    ----------
    npz_path : Path to the npz file
    batch_size : Batch size
    shuffle : Whether to shuffle (only for training)
    num_workers : Number of PyTorch DataLoader workers
    window_size : Size of each window segment
    overlap : Overlap between consecutive windows
    dtype : Data type for tensor conversion
    transform : Optional data augmentation function
    
    Returns
    -------
    DataLoader : PyTorch DataLoader with samples shaped (batch, features, window_size)
    """
    dataset = SHLMagDataset(
        npz_path=npz_path,
        window_size=window_size,
        overlap=overlap,
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

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y



# ----------------------------------------------------------------------------- # 
# -------------------- Dataset for SHL NPZ files -------------------- #
# ----------------------------------------------------------------------------- #
class SHLNpzDataset(Dataset):
    """
    Dataset that loads preprocessed NPZ files containing SHL sensor data.
    
    Each sample:
        X : (window_size, features) torch.float32
        y : int (class label)
    """
    
    def __init__(self, 
                 npz_path: str,
                 window_size: int = 500,
                 overlap: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 transform=None):
        """
        Initialize the dataset from a npz file.
        
        Parameters
        ----------
        npz_path: str
            Path to the npz file
        window_size: int
            Size of each window segment
        overlap: float
            Overlap between consecutive windows (0.0 to 1.0)
        dtype: torch.dtype
            Data type for tensor conversion
        transform: callable
            Optional transform to be applied on a sample
        """
        data = np.load(npz_path)
        x_data = data['data']  # (n_samples, n_features)
        y_data = data['labels']  # (n_samples,)
        
        # 计算窗口步长和数量（矢量化分窗）
        step = int(window_size * (1 - overlap))
        n_samples, n_features = x_data.shape
        n_windows = (n_samples - window_size) // step + 1
        
        if n_windows <= 0:
            raise ValueError(f"Window size {window_size} is too large for data with {n_samples} samples")
        
        # 创建窗口索引
        indices = np.arange(window_size)[None, :] + step * np.arange(n_windows)[:, None]
        
        # 矢量化分窗
        self.x = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
        for i in range(n_features):
            self.x[:, :, i] = x_data[indices, i]
        
        # 对应窗口的标签
        self.y = y_data[indices[:, 0]]
        
        self.dtype = dtype
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=self.dtype)
        y = torch.as_tensor(self.y[idx], dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def get_npz_dataloader(npz_path: str,
                       batch_size: int = 32,
                       shuffle: bool = True,
                       num_workers: int = 4,
                       window_size: int = 500,
                       overlap: float = 0.0,
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
    window_size: int
        Size of each window segment
    overlap: float
        Overlap between consecutive windows
    dtype: torch.dtype
        Data type for tensor conversion
    transform: callable
        Optional transform to be applied on a sample
    """
    dataset = SHLNpzDataset(
        npz_path=npz_path,
        window_size=window_size,
        overlap=overlap,
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
        window_size=500,
        overlap=0.0
        )
    
    # process and save the dataset
    process_and_save_dataset(
        cfg=cfg,
        output_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data",
        train_ratio=0.7
    )
    
    # load the dataset using DataLoader
    train_loader = get_npz_dataloader(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_train.npz",
        batch_size=1024,
        shuffle=True
    )


    test_loader = get_npz_dataloader(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_validation.npz",
        batch_size=1024,
        shuffle=False
    )
    
    # debug
    x_train, y_train = next(iter(train_loader))
    print("Training batch shape:", x_train.shape)  #  (batch_size, features, window_size)
    print("Training batch y shape:", y_train.shape)  #  (batch_size,)
    print("Train Labels:", y_train.unique())
    x_test, y_test = next(iter(test_loader))
    print("Test batch shape:", x_test.shape)  #  (batch_size, features, window_size)
    print("Test batch y shape:", y_test.shape)  #  (batch_size,)
    print("Test Labels:", y_test.unique())
