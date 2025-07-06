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


def get_dataset(
    csv_cls: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/preprocessing/geoMagDataset/class3.csv",
    npz_path: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset.npz",
    test_ratio: float = 0.30,
    random_state: int = 42,
):
    # 1. Load data and shuffle ------------------------------------------------
    df = pd.read_csv(csv_cls, header=None).sample(frac=1, random_state=random_state)
    df = df[0].str.split(";", expand=True)         # split into 622 columns
    y = df[621]                                    # column 622 is the label
    X = df.drop(columns=[621]).astype(float)       # retain 0–620 => 621 features

    print("— Original data shape —")
    print(f"X: {X.shape} | y: {y.shape}")          # (376, 621)

    # 2. Min-Max scaling ------------------------------------------------------
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Stratified train/test split -----------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled,
        y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train X: {X_tr.shape} | Test X: {X_te.shape}")  # (263, 621)/(113, 621)

    # 4. Integer label encoding ----------------------------------------------
    le = LabelEncoder()
    y_tr_int = le.fit_transform(y_tr)
    y_te_int = le.transform(y_te)

    # 5. SMOTE oversampling (train set only) ----------------------------------
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr_int)

    print("SMOTE class distribution:", np.bincount(y_tr_res))  # [207 207 207]

    # 6. Save to NPZ ----------------------------------------------------------
    np.savez_compressed(
        npz_path,
        X_train=X_tr_res.astype(np.float32),
        y_train=y_tr_res.astype(np.int64),
        X_test=X_te.astype(np.float32),
        y_test=y_te_int.astype(np.int64),
        classes=le.classes_,
    )
    print(f"Data saved to {npz_path}")

    return X_tr_res, y_tr_res, X_te, y_te_int, le

# -------------------------- image-conversion utils -------------------------- #
# Color lookup table (BGR order → later np.flip for RGB) — values 0-255
COLORS = {
    0b000: (255,   0,   0),   # Blue
    0b001: (255,   0, 255),   # Purple  (X only)
    0b010: (  0,   0, 255),   # Red     (Y only)
    0b100: (  0, 255,   0),   # Green   (Z only)
    0b011: (  0,   0,   0),   # Black   (X+Y)
    0b101: (  0, 128, 128),   # Olive   (X+Z)
    0b110: (255, 255,   0),   # Turquoise (Y+Z)
    0b111: (255, 255, 255),   # White   (X+Y+Z)
}

def signal_to_image(vec621: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """
    Convert flattened 621-dim vector -> 216×216×3 RGB uint8 image.
    A column's color is chosen by which axes (X,Y,Z) lie *under the curve*
    at that timestep.
    Args
    ----
    vec621 : (621,)  flattened sequence  x0,y0,z0,x1,...
    thresh : float   threshold to decide “under curve”. 0 works because
                     signals are Min-Max scaled to [0,1].
    Returns
    -------
    img : (216,216,3)  uint8 RGB
    """
    seq = vec621.reshape(207, 3)          # (T,3)  -> time major
    img  = np.zeros((216, 216, 3), dtype=np.uint8)

    for t in range(207):
        x, y, z = seq[t]
        mask = ((x > thresh) << 0) | ((y > thresh) << 1) | ((z > thresh) << 2)
        color = COLORS[mask]
        img[:, t, :] = color[::-1]        # convert BGR→RGB when assigning

    # pad rightmost 9 columns with last color
    img[:, 207:, :] = img[:, 206:207, :]
    return img


# ------------------------------- main pipeline ------------------------------ #
def get_imageDataset(
    csv_cls: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/preprocessing/geoMagDataset/class3.csv",
    npz_path: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset.npz",
    npz_img_path: str = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset_img.npz",
    test_ratio: float = 0.30,
    random_state: int = 42,
):
    # 1 ─ load & shuffle ------------------------------------------------------
    df = pd.read_csv(csv_cls, header=None).sample(frac=1, random_state=random_state)
    df = df[0].str.split(";", expand=True)
    y = df[621]
    X = df.drop(columns=[621]).astype(float)

    print("— Original data shape —")
    print(f"X: {X.shape} | y: {y.shape}")          # (376, 621)

    # 2 ─ Min-Max scaling -----------------------------------------------------
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3 ─ stratified split ----------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_ratio,
        random_state=random_state, stratify=y
    )
    print(f"Train X: {X_tr.shape} | Test X: {X_te.shape}")

    # 4 ─ label encoding ------------------------------------------------------
    le = LabelEncoder()
    y_tr_int = le.fit_transform(y_tr)
    y_te_int = le.transform(y_te)

    # 5 ─ SMOTE (train set only) ---------------------------------------------
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr_int)
    print("SMOTE class distribution:", np.bincount(y_tr_res))

    # 6 ─ save numeric dataset -----------------------------------------------
    np.savez_compressed(
        npz_path,
        X_train=X_tr_res.astype(np.float32),
        y_train=y_tr_res.astype(np.int64),
        X_test=X_te.astype(np.float32),
        y_test=y_te_int.astype(np.int64),
        classes=le.classes_,
    )
    print(f"Numeric data saved to {npz_path}")

    # ---------------- NEW: image conversion ---------------- #
    print("Converting signals to RGB images …")
    X_tr_img = np.stack([signal_to_image(v) for v in X_tr_res], axis=0)
    X_te_img = np.stack([signal_to_image(v) for v in X_te],     axis=0)

    # 7 ─ save image dataset -----------------------------------------------
    np.savez_compressed(
        npz_img_path,
        X_train_img=X_tr_img,    # uint8
        y_train=y_tr_res.astype(np.int64),
        X_test_img=X_te_img,
        y_test=y_te_int.astype(np.int64),
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


# 在 if __name__ == "__main__": 部分添加可视化调用
if __name__ == "__main__":
    # 生成数据集
    # get_dataset()
    # get_imageDataset()
    
    # 可视化样本
    print("\n" + "="*60)
    print("VISUALIZING CLASS SAMPLES")
    print("="*60)
    
    # 展示每个类别的代表样本
    visualize_class_samples()
    
    # 展示每个类别的多个样本
    visualize_multiple_samples_per_class(samples_per_class=3)
