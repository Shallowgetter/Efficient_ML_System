# ori_getDataset.py
# -*- coding: utf-8 -*-
"""
ORI moving-target radar dataset preprocessing + sequence slicing + NPZ persistency
The pipeline mirrors getDataset.py style but focuses on sequence data only.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
#                          Helper: Slice One Signature
# -----------------------------------------------------------------------------
def _slice_signature_to_sequences(sig_array,
                                  seq_len: int = 10,
                                  stride: int | None = None,
                                  drop_last: bool = True):
    """
    Convert one spectrogram (B × T) into fixed-length sequences.

    Parameters
    ----------
    sig_array : np.ndarray
        Spectrogram magnitude (B, T) after log-scale.
    seq_len : int
        Number of consecutive spectra per sequence.
    stride : int | None
        Step size between windows; defaults to seq_len (non-overlapping).
    drop_last : bool
        If True, discard trailing frames < seq_len; else pad with zeros.

    Returns
    -------
    list[np.ndarray]  (n_seq, seq_len, B)
    """
    if stride is None:
        stride = seq_len

    B, T = sig_array.shape
    sequences = []
    for start in range(0, T - seq_len + 1, stride):
        seq = sig_array[:, start : start + seq_len]          # (B, seq_len)
        sequences.append(seq.T)                              # → (seq_len, B)

    # handle residual part
    residual = T % stride
    if (not drop_last) and residual and residual < seq_len:
        pad_width = seq_len - residual
        pad = np.zeros((B, pad_width), dtype=sig_array.dtype)
        seq = np.concatenate([sig_array[:, -residual:], pad], axis=1).T
        sequences.append(seq)

    return sequences


# -----------------------------------------------------------------------------
#                     Helper: ORI Dataset Preparation Pipeline
# -----------------------------------------------------------------------------
def _prepare_ori_dataset(
    npy_file: str,
    seq_len: int,
    test_ratio: float,
    random_state: int,
    drop_last: bool = True,
):
    """
    Load ORI dataset, perform track-level split, sequence slicing,
    per-sequence min-max scaling, and label encoding.

    Returns
    -------
    X_train_seq, y_train_int, X_test_seq, y_test_int, label_encoder
    """
    # 1 ─ Load raw .npy signatures ------------------------------------------
    signatures = np.load(npy_file, allow_pickle=True)
    print(f"[INFO] Loaded {len(signatures)} tracks from {npy_file}")

    # 2 ─ Track-level train/test split --------------------------------------
    all_indices = list(range(len(signatures)))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=test_ratio, random_state=random_state
    )
    print(f"[INFO] Train tracks: {len(train_idx)} | Test tracks: {len(test_idx)}")

    # 3 ─ Collect class names & encode --------------------------------------
    class_names = sorted({sig["class_name"] for sig in signatures})
    le = LabelEncoder()
    le.fit(class_names)
    print(f"[INFO] Classes: {le.classes_.tolist()}")

    # 4 ─ Iterate tracks → sequences ----------------------------------------
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    def _process_track(sig, container_X, container_y):
        # 4.1 Log-magnitude & transpose (B × T)
        arr = 20.0 * np.log10(np.abs(sig["signature"])).astype(np.float32).T
        # 4.2 Slice into fixed-length sequences
        seqs = _slice_signature_to_sequences(
            arr, seq_len=seq_len, stride=seq_len, drop_last=drop_last
        )
        # 4.3 Per-sequence min-max scaling to [0,1]
        for seq in seqs:
            _min = seq.min()
            _range = max(seq.max() - _min, 1e-6)
            seq_norm = (seq - _min) / _range
            container_X.append(seq_norm.astype(np.float32))
            container_y.append(le.transform([sig["class_name"]])[0])

    for idx in train_idx:
        _process_track(signatures[idx], X_train, y_train)

    for idx in test_idx:
        _process_track(signatures[idx], X_test, y_test)

    print(f"[INFO] Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")

    # 5 ─ Convert to np.ndarray ---------------------------------------------
    X_train_arr = np.stack(X_train, axis=0)  # (N_train, seq_len, B)
    X_test_arr  = np.stack(X_test,  axis=0)  # (N_test,  seq_len, B)

    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr  = np.asarray(y_test,  dtype=np.int64)

    return X_train_arr, y_train_arr, X_test_arr, y_test_arr, le


# -----------------------------------------------------------------------------
#                          Public API: get_dataset
# -----------------------------------------------------------------------------
def get_dataset(
    npy_file: str = "moving_target_dataset.npy",
    npz_out: str = "oriDataset_seq.npz",
    seq_len: int = 10,
    test_ratio: float = 0.20,
    random_state: int = 42,
    drop_last: bool = True,
    visualize: bool = True,
    vis_save_path: str = "class_samples_visualization.png"
):
    """
    ORI dataset preprocessing entry point.

    Parameters
    ----------
    npy_file : str
        Path to the raw 'moving_target_dataset.npy'.
    npz_out : str
        Output path for compressed NPZ.
    seq_len : int
        Number of consecutive spectra per sequence.
    test_ratio : float
        Fraction of tracks reserved for testing.
    random_state : int
        RNG seed ensuring reproducibility.
    drop_last : bool
        Whether to discard residual frames shorter than seq_len.
    visualize : bool
        Whether to create visualization of sample sequences.
    vis_save_path : str
        Path to save visualization image.

    Returns
    -------
    X_train, y_train, X_test, y_test, label_encoder
    """
    X_tr, y_tr, X_te, y_te, le = _prepare_ori_dataset(
        npy_file, seq_len, test_ratio, random_state, drop_last
    )

    # Print dataset statistics
    print_dataset_statistics(X_tr, y_tr, X_te, y_te, le)

    # 6 ─ Save to NPZ --------------------------------------------------------
    np.savez_compressed(
        npz_out,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        classes=le.classes_,
    )
    print(f"[INFO] Sequence dataset saved to {npz_out}")

    # 7 ─ Visualization (optional) ------------------------------------------
    if visualize:
        visualize_samples_per_class(X_tr, y_tr, X_te, y_te, le, vis_save_path)

    return X_tr, y_tr, X_te, y_te, le


# -----------------------------------------------------------------------------
#                          Visualization Function
# -----------------------------------------------------------------------------
def visualize_samples_per_class(
    X_train, y_train, X_test, y_test, le,
    save_path: str = "class_samples_visualization.png",
    figsize: tuple = (15, 10)
):
    """
    Visualize one sample sequence for each class
    
    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and testing sequence data (N, seq_len, B)
    y_train, y_test : np.ndarray
        Training and testing labels
    le : LabelEncoder
        Label encoder
    save_path : str
        Path to save the visualization image
    figsize : tuple
        Figure size
    """
    # Combine training and testing data for more sample choices
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    
    classes = le.classes_
    n_classes = len(classes)
    
    # Calculate subplot layout
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    plt.suptitle('Sample Sequence Visualization for Each Class', fontsize=16, fontweight='bold')
    
    for i, class_name in enumerate(classes):
        # Find the first sample of this class
        class_indices = np.where(y_all == i)[0]
        if len(class_indices) == 0:
            continue
            
        sample_idx = class_indices[0]
        sample_seq = X_all[sample_idx]  # (seq_len, B)
        
        # Calculate subplot position
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Create heatmap
        im = ax.imshow(sample_seq.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'Class: {class_name}\nSequence Shape: {sample_seq.shape}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Frequency Bins')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide extra subplots
    for i in range(n_classes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Class sample visualization saved to: {save_path}")
    plt.show()


def print_dataset_statistics(X_train, y_train, X_test, y_test, le):
    """
    Print dataset statistics
    """
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    print(f"Training set: {X_train.shape[0]} sequences")
    print(f"Testing set: {X_test.shape[0]} sequences")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Frequency bins: {X_train.shape[2]}")
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    
    print(f"\nClass distribution:")
    y_all = np.concatenate([y_train, y_test])
    for i, class_name in enumerate(le.classes_):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        total_count = train_count + test_count
        print(f"  {class_name}: {total_count} sequences (train: {train_count}, test: {test_count})")
    
    print("="*60)


# -----------------------------------------------------------------------------
#                                CLI Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    get_dataset(
        npy_file="/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/Open_Radar/moving_target_dataset.npy",
        npz_out="oriDataset_seq.npz",
        seq_len=10,
        test_ratio=0.20,
        random_state=42,
        visualize=True,
        vis_save_path="radar_dataset_samples.png"
    )

