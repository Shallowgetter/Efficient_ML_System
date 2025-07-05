"""
This file visualizes the GeoMagDataset for 3-axis mag data.
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_geo_mag_data(data, labels, classes, title="GeoMagnetic Data Visualization", num_samples=5):
    """
    可视化地磁数据的时序图
    
    Args:
        data: 特征数据 (n_samples, n_features)
        labels: 标签数据 (n_samples,)
        classes: 类别名称
        title: 图标题
        num_samples: 要可视化的样本数量
    """
    # 假设特征维度可以重新组织为三轴时序数据
    # 根据原始数据结构，每个样本应该包含时序信息
    n_features = data.shape[1]
    
    # 假设数据是按时间步展开的三轴数据 (time_steps * 3)
    # 如果不是，需要根据实际数据结构调整
    if n_features % 3 == 0:
        time_steps = n_features // 3
        
        # 选择不同类别的样本进行可视化
        unique_labels = np.unique(labels)
        samples_to_plot = []
        
        for label in unique_labels[:num_samples]:
            idx = np.where(labels == label)[0]
            if len(idx) > 0:
                samples_to_plot.append((data[idx[0]], label))
        
        # 创建子图
        fig, axes = plt.subplots(len(samples_to_plot), 3, figsize=(15, 4 * len(samples_to_plot)))
        if len(samples_to_plot) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (sample_data, label) in enumerate(samples_to_plot):
            # 重新组织数据为三轴时序
            reshaped_data = sample_data.reshape(time_steps, 3)
            
            for axis in range(3):
                axes[i, axis].plot(reshaped_data[:, axis], label=f"Axis {axis + 1}")
                axes[i, axis].set_title(f"{title} - {classes[label]} - Axis {axis + 1}")
                axes[i, axis].set_xlabel("Time Steps")
                axes[i, axis].set_ylabel("Magnetic Field Strength")
                axes[i, axis].legend()
                axes[i, axis].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Warning: Feature dimension {n_features} is not divisible by 3")
        print("Cannot reshape into 3-axis time series data")

def visualize_feature_distribution(data, labels, classes, title="Feature Distribution"):
    """
    可视化特征分布
    """
    plt.figure(figsize=(12, 8))
    
    # 选择前几个特征进行可视化
    n_features_to_plot = min(9, data.shape[1])
    
    for i in range(n_features_to_plot):
        plt.subplot(3, 3, i + 1)
        
        for label in np.unique(labels):
            mask = labels == label
            plt.hist(data[mask, i], alpha=0.6, label=classes[label], bins=20)
        
        plt.title(f"Feature {i + 1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 主要执行代码
npz_path = "data/geoMag/geoMagDataset.npz"

# Load the dataset with allow_pickle=True to load object arrays
data = np.load(npz_path, allow_pickle=True)
X_train_res = data['X_train']
y_train_res = data['y_train']
X_test_np = data['X_test']
y_test_int = data['y_test']
classes = data['classes']

print(f"Training data shape: {X_train_res.shape}")
print(f"Test data shape: {X_test_np.shape}")
print(f"Classes: {classes}")
print(f"Unique training labels: {np.unique(y_train_res)}")

# 可视化训练数据的时序图
visualize_geo_mag_data(X_train_res, y_train_res, classes, title="Training Data Time Series", num_samples=3)

# 可视化测试数据的时序图
visualize_geo_mag_data(X_test_np, y_test_int, classes, title="Test Data Time Series", num_samples=3)

# 可视化特征分布
visualize_feature_distribution(X_train_res, y_train_res, classes, title="Training Data Feature Distribution")