# -*- coding: utf-8 -*-
"""
Geophysical magnetic sensor dataset preprocessing + SMOTE oversampling + NPZ persistency
Steps:
1) Load class3.csv (621 features)
2) Stratified 70/30 split for train/test
3) Apply SMOTE(k=5) only on the training set to balance classes to [207, 207, 207]
4) Save to geoMagDataset.npz for downstream DataLoader
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    raise ImportError(
        "imbalanced-learn not found. Please install it via: pip install imbalanced-learn"
    ) from e


# -----------------------------------------------------------------------------
#                           Helper: Numeric Pipeline
# -----------------------------------------------------------------------------
def _prepare_numeric_dataset(csv_cls, test_ratio, random_state):
    """
    Load raw CSV, Min-Max scale, stratified split, label-encode, SMOTE.
    Returns:
        X_tr_res (np.float32), y_tr_res (np.int64),
        X_te     (np.float32), y_te_int (np.int64),
        le       (LabelEncoder fitted on training labels)
    """
    # 1. Load and shuffle
    df = pd.read_csv(csv_cls, header=None).sample(frac=1, random_state=random_state)
    df = df[0].str.split(";", expand=True)
    y = df[621]
    X = df.drop(columns=[621]).astype(float)
    print("— Original data shape —")
    print(f"X: {X.shape} | y: {y.shape}")

    # 2. Min-Max scaling
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Stratified train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train X: {X_tr.shape} | Test X: {X_te.shape}")

    # 4. Integer label encoding
    le = LabelEncoder()
    y_tr_int = le.fit_transform(y_tr)
    y_te_int = le.transform(y_te)

    # 5. SMOTE oversampling (train only)
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr_int)
    print("SMOTE class distribution:", np.bincount(y_tr_res))

    return (
        X_tr_res.astype(np.float32),
        y_tr_res.astype(np.int64),
        X_te.astype(np.float32),
        y_te_int.astype(np.int64),
        le,
    )

# -----------------------------------------------------------------------------
#                      Signal-to-Image Conversion
# -----------------------------------------------------------------------------
# Color lookup table: mask (0–7) → (B, G, R)
_COLORS = {
    0: (255,   0,   0),   # background (Blue)
    1: (255,   0, 255),   # X only     (Purple)
    2: (  0,   0, 255),   # Y only     (Red)
    3: (  0,   0,   0),   # X+Y        (Black)
    4: (  0, 255,   0),   # Z only     (Green)
    5: (  0, 128, 128),   # X+Z        (Olive)
    6: (255, 255,   0),   # Y+Z        (Turquoise)
    7: (255, 255, 255),   # X+Y+Z      (White)
}

def signal_to_image(vec621: np.ndarray, height: int = 216) -> np.ndarray:
    """
    Convert a 621-dim flattened signal (207×3) into a (height×216×3) RGB image
    by filling the area under each curve (X, Y, Z) with distinct colors.
    """
    seq = vec621.reshape(207, 3)
    width = 207
    img = np.zeros((height, 216, 3), dtype=np.uint8)

    # compute pixel heights per axis per time step
    pixel_heights = (seq * (height - 1)).round().astype(int)

    # build a mask image: each pixel stores bits for X(1), Y(2), Z(4)
    mask = np.zeros((height, width), dtype=np.uint8)
    for t in range(width):
        for axis in range(3):
            h = pixel_heights[t, axis]
            start_row = height - 1 - h
            mask[start_row:, t] |= (1 << axis)

    # map mask→color
    for m_val, bgr in _COLORS.items():
        ys, xs = np.where(mask == m_val)
        if ys.size > 0:
            # assign RGB by reversing BGR
            img[ys, xs, 0] = bgr[2]
            img[ys, xs, 1] = bgr[1]
            img[ys, xs, 2] = bgr[0]

    # pad rightmost columns
    img[:, 207:, :] = img[:, 206:207, :]
    return img


# ------------------------------- main pipeline ------------------------------ #
# -----------------------------------------------------------------------------
#                          Public API: Numeric Dataset
# -----------------------------------------------------------------------------
def get_dataset(
    csv_cls: str,
    npz_path: str,
    test_ratio: float = 0.30,
    random_state: int = 42,
):
    """
    Generate and save numeric dataset.
    Inputs/outputs unchanged from original.
    Returns: X_tr_res, y_tr_res, X_te, y_te_int, label_encoder
    """
    X_tr_res, y_tr_res, X_te, y_te_int, le = _prepare_numeric_dataset(
        csv_cls, test_ratio, random_state
    )
    np.savez_compressed(
        npz_path,
        X_train=X_tr_res,
        y_train=y_tr_res,
        X_test=X_te,
        y_test=y_te_int,
        classes=le.classes_,
    )
    print(f"Data saved to {npz_path}")
    return X_tr_res, y_tr_res, X_te, y_te_int, le

# -----------------------------------------------------------------------------
#                         Public API: Image Dataset
# -----------------------------------------------------------------------------
def get_imageDataset(
    csv_cls: str,
    npz_path: str,
    npz_img_path: str,
    test_ratio: float = 0.30,
    random_state: int = 42,
):
    """
    Generate and save both numeric and image datasets.
    Inputs/outputs unchanged from original.
    Returns:
      (X_tr_res, y_tr_res, X_te, y_te_int, label_encoder),
      (X_tr_img, X_te_img)
    """
    X_tr_res, y_tr_res, X_te, y_te_int, le = _prepare_numeric_dataset(
        csv_cls, test_ratio, random_state
    )

    print("Converting signals to filled-area RGB images …")
    X_tr_img = np.stack([signal_to_image(v) for v in X_tr_res], axis=0)
    X_te_img = np.stack([signal_to_image(v) for v in X_te], axis=0)

    np.savez_compressed(
        npz_img_path,
        X_train_img=X_tr_img,
        y_train=y_tr_res,
        X_test_img=X_te_img,
        y_test=y_te_int,
        classes=le.classes_,
    )
    print(f"Image data saved to {npz_img_path}")

    return (X_tr_res, y_tr_res, X_te, y_te_int, le), (X_tr_img, X_te_img)


def visualize_class_samples():
    """
    可视化每个类别的图像样本
    """
    # 加载图像数据
    img_npz_path = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset_img.npz"
    
    if not os.path.exists(img_npz_path):
        print(f"Image dataset not found at {img_npz_path}")
        print("Running get_imageDataset() to generate it...")
        get_imageDataset()
    
    # 加载数据 - 添加 allow_pickle=True
    data = np.load(img_npz_path, allow_pickle=True)
    X_train_img = data["X_train_img"]  # (621, 216, 216, 3)
    y_train = data["y_train"]
    X_test_img = data["X_test_img"]    # (113, 216, 216, 3)
    y_test = data["y_test"]
    class_names = data["classes"]      # ['Heavy', 'Light', 'Medium']
    
    print(f"Train images: {X_train_img.shape}")
    print(f"Test images: {X_test_img.shape}")
    print(f"Classes: {class_names}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Geomagnetic Signal Images by Class', fontsize=16, fontweight='bold')
    
    # 为每个类别找到第一个样本并可视化
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        
        # 在训练集中找到该类别的第一个样本
        train_indices = np.where(y_train == class_idx)[0]
        if len(train_indices) > 0:
            sample_img = X_train_img[train_indices[0]]
            axes[0, class_idx].imshow(sample_img)
            axes[0, class_idx].set_title(f'Train - {class_name}\n(Sample {train_indices[0]})', 
                                       fontsize=12, fontweight='bold')
            axes[0, class_idx].axis('off')
            
            # 显示类别统计
            train_count = len(train_indices)
            axes[0, class_idx].text(0.02, 0.98, f'Count: {train_count}', 
                                  transform=axes[0, class_idx].transAxes,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                  fontsize=10, verticalalignment='top')
        
        # 在测试集中找到该类别的第一个样本
        test_indices = np.where(y_test == class_idx)[0]
        if len(test_indices) > 0:
            sample_img = X_test_img[test_indices[0]]
            axes[1, class_idx].imshow(sample_img)
            axes[1, class_idx].set_title(f'Test - {class_name}\n(Sample {test_indices[0]})', 
                                       fontsize=12, fontweight='bold')
            axes[1, class_idx].axis('off')
            
            # 显示类别统计
            test_count = len(test_indices)
            axes[1, class_idx].text(0.02, 0.98, f'Count: {test_count}', 
                                   transform=axes[1, class_idx].transAxes,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                   fontsize=10, verticalalignment='top')
    
    # 添加行标签
    axes[0, 0].text(-0.1, 0.5, 'Training Set', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, 
                   verticalalignment='center', horizontalalignment='right')
    axes[1, 0].text(-0.1, 0.5, 'Test Set', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, 
                   verticalalignment='center', horizontalalignment='right')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/class_samples_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # 打印详细统计信息
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print("\nTraining Set:")
    for class_idx in range(len(class_names)):
        count = np.sum(y_train == class_idx)
        percentage = count / len(y_train) * 100
        print(f"  {class_names[class_idx]}: {count} samples ({percentage:.1f}%)")
    
    print("\nTest Set:")
    for class_idx in range(len(class_names)):
        count = np.sum(y_test == class_idx)
        percentage = count / len(y_test) * 100
        print(f"  {class_names[class_idx]}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nTotal: {len(y_train)} training + {len(y_test)} test = {len(y_train) + len(y_test)} samples")


def visualize_multiple_samples_per_class(samples_per_class=3):
    """
    可视化每个类别的多个样本
    """
    # 加载图像数据
    img_npz_path = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset_img.npz"
    
    if not os.path.exists(img_npz_path):
        print(f"Image dataset not found at {img_npz_path}")
        print("Running get_imageDataset() to generate it...")
        get_imageDataset()
    
    # 添加 allow_pickle=True
    data = np.load(img_npz_path, allow_pickle=True)
    X_train_img = data["X_train_img"]
    y_train = data["y_train"]
    class_names = data["classes"]
    
    # 创建大图
    fig, axes = plt.subplots(len(class_names), samples_per_class, 
                            figsize=(samples_per_class * 4, len(class_names) * 4))
    fig.suptitle(f'Multiple Samples per Class ({samples_per_class} samples each)', 
                fontsize=16, fontweight='bold')
    
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        class_indices = np.where(y_train == class_idx)[0]
        
        for sample_idx in range(min(samples_per_class, len(class_indices))):
            img_idx = class_indices[sample_idx]
            sample_img = X_train_img[img_idx]
            
            row, col = class_idx, sample_idx
            axes[row, col].imshow(sample_img)
            axes[row, col].set_title(f'{class_name} - Sample {sample_idx + 1}', 
                                   fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = f"/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/multiple_samples_per_class_{samples_per_class}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multiple samples visualization saved to: {save_path}")
    
    plt.show()

# -----------------------------------------------------------------------------
#                          Public API: Feature-Extractred Dataset
# -----------------------------------------------------------------------------
def get_dataset_FE(
    fe_csv: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/preprocessing/geoMagDataset/class3_FE.csv",
    raw_csv: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/preprocessing/geoMagDataset/class3.csv",
    npz_path: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset_FE.npz",
    test_ratio: float = 0.30,
    random_state: int = 42,
    drop_cols: tuple = (44, 45, 46, 47, 48, 49),
):
    """
    Feature-engineered dataset pipeline analogous to get_dataset().

    Parameters
    ----------
    fe_csv : str
        Path to 'class3_FE.csv'.
    raw_csv : str
        Path to raw 'class3.csv' (used only for labels).
    npz_path : str
        Output path for compressed NPZ.
    test_ratio : float
        Fraction of data reserved for testing.
    random_state : int
        RNG seed ensuring reproducibility.
    drop_cols : tuple
        Column indices (0-based) to be removed from FE features.

    Returns
    -------
    X_train_res, y_train_res, X_test, y_test_int, label_encoder
    """
    # 1 ─ Load FE features and corresponding labels --------------------------
    df_fe = pd.read_csv(fe_csv, header=None).drop(columns=list(drop_cols))
    df_label = (
        pd.read_csv(raw_csv, header=None)[0]
        .str.split(";", expand=True)[621]
        .astype(str)
    )

    df = pd.concat([df_fe, df_label], axis=1, join="inner")
    df = df.sample(frac=1, random_state=random_state)  # shuffle rows

    y = df[621]
    X = df.drop(columns=[621]).astype(float).fillna(0.0)

    print("— Original FE data shape —")
    print(f"X: {X.shape} | y: {y.shape}")

    # 2 ─ Min-Max scaling -----------------------------------------------------
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3 ─ Stratified train/test split ----------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train X: {X_tr.shape} | Test X: {X_te.shape}")

    # 4 ─ Integer label encoding ---------------------------------------------
    le = LabelEncoder()
    y_tr_int = le.fit_transform(y_tr)
    y_te_int = le.transform(y_te)

    # 5 ─ SMOTE oversampling (train set only) ---------------------------------
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr_int)
    print("SMOTE class distribution:", np.bincount(y_tr_res))

    # 6 ─ Save to NPZ ---------------------------------------------------------
    np.savez_compressed(
        npz_path,
        X_train=X_tr_res.astype(np.float32),
        y_train=y_tr_res.astype(np.int64),
        X_test=X_te.astype(np.float32),
        y_test=y_te_int.astype(np.int64),
        classes=le.classes_,
    )
    print(f"FE dataset saved to {npz_path}")

    return X_tr_res, y_tr_res, X_te, y_te_int, le



# 在 if __name__ == "__main__": 部分添加可视化调用
if __name__ == "__main__":
    # 生成数据集
    # get_dataset()
    # get_dataset_FE()
    get_imageDataset(
        csv_cls="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/preprocessing/geoMagDataset/class3.csv",
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset.npz",
        npz_img_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset_img.npz",
        test_ratio=0.30,
        random_state=42,
    )
    
    # 可视化样本
    print("\n" + "="*60)
    print("VISUALIZING CLASS SAMPLES")
    print("="*60)
    
    # 展示每个类别的代表样本
    visualize_class_samples()
    
    # 展示每个类别的多个样本
    visualize_multiple_samples_per_class(samples_per_class=3)
