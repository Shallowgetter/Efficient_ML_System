# coding: utf-8
"""
Visualise six time-series features across multiple CSV samples.
All comments and print statements are in English as required.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------------
# 1. Parameters
# ------------------------------------------------------------------
DATA_DIR = r"/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/MSV/Vehicles_magnetic_signatures"      
FEATURES  = ["x1", "y1", "z1", "x2", "y2", "z2"]

# 创建分析目录
ANALYSIS_DIR = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

FIG_PATH = os.path.join(ANALYSIS_DIR, "multi_sample_features.png")
REPORT_PATH = os.path.join(ANALYSIS_DIR, "dataset_analysis_report.txt")

# ------------------------------------------------------------------
# 指定要分析的文件列表 (替换随机抽样)
# ------------------------------------------------------------------
# 方式1: 直接在代码中指定文件名
SPECIFIED_FILES = [
    "10500001-sign.csv", # 4.0 m
    "11600001-sign.csv", # 15.57 m
    "11600033-sign.csv", # 7.35 m
]

# 方式2: 从文本文件读取文件名列表 
FILE_LIST_PATH = os.path.join(ANALYSIS_DIR, "analysis_file_list.txt")

def load_specified_files():
    """
    加载指定的文件列表
    优先级: 1. 文本文件列表 2. 代码中的SPECIFIED_FILES
    """
    files_to_analyze = []
    
    # 首先尝试从文本文件读取
    if os.path.exists(FILE_LIST_PATH):
        print(f"Loading file list from: {FILE_LIST_PATH}")
        try:
            with open(FILE_LIST_PATH, 'r', encoding='utf-8') as f:
                files_to_analyze = [line.strip() for line in f.readlines() 
                                  if line.strip() and not line.startswith('#')]
            print(f"Loaded {len(files_to_analyze)} files from text file.")
        except Exception as e:
            print(f"Error reading file list: {e}")
            print("Using files specified in code instead.")
            files_to_analyze = SPECIFIED_FILES
    else:
        # 如果文本文件不存在，使用代码中指定的文件
        print("File list not found, using files specified in code.")
        files_to_analyze = SPECIFIED_FILES
        
        # 创建示例文件列表文件
        print(f"Creating example file list at: {FILE_LIST_PATH}")
        with open(FILE_LIST_PATH, 'w', encoding='utf-8') as f:
            f.write("# Analysis File List\n")
            f.write("# Add one CSV filename per line\n")
            f.write("# Lines starting with # are comments\n")
            f.write("# Example files:\n")
            for fname in SPECIFIED_FILES:
                f.write(f"{fname}\n")
        print("You can edit this file to specify different files for analysis.")
    
    return files_to_analyze

def validate_files(file_list):
    """
    验证指定的文件是否存在
    """
    # 获取数据目录中所有可用的CSV文件
    available_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    available_files_set = set(available_files)
    
    valid_files = []
    missing_files = []
    
    for fname in file_list:
        if fname in available_files_set:
            valid_files.append(fname)
        else:
            missing_files.append(fname)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} specified files not found:")
        for fname in missing_files:
            print(f"  - {fname}")
    
    print(f"Found {len(valid_files)} valid files out of {len(file_list)} specified.")
    
    if not valid_files:
        print("No valid files found! Available files in directory:")
        for fname in available_files[:10]:  # 显示前10个文件作为示例
            print(f"  - {fname}")
        if len(available_files) > 10:
            print(f"  ... and {len(available_files) - 10} more files")
    
    return valid_files, missing_files

# 加载指定的文件列表
specified_files = load_specified_files()
print(f"Files specified for analysis: {len(specified_files)}")

# 验证文件是否存在
valid_files, missing_files = validate_files(specified_files)

if not valid_files:
    print("ERROR: No valid files to analyze. Please check your file list.")
    exit(1)

# 使用验证过的文件列表
selected_files = valid_files

# 记录分析开始时间
import datetime
start_time = datetime.datetime.now()

# ------------------------------------------------------------------
# 2. Load data from specified CSV files
# ------------------------------------------------------------------
col_data = {f: [] for f in FEATURES}
file_names = []
file_info = []

print(f"Loading {len(selected_files)} specified files...")

# 加载指定的文件
for file in selected_files:
    fpath = os.path.join(DATA_DIR, file)
    try:
        df = pd.read_csv(fpath)
        original_shape = df.shape
        
        # 检查是否包含所需特征
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            print(f"Skip {file}: missing features {missing_features}")
            file_info.append({
                'filename': file,
                'status': 'skipped',
                'reason': f'missing features: {missing_features}',
                'shape': original_shape
            })
            continue
            
        df_clean = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        df_clean.dropna(how="all", inplace=True)
        
        if df_clean.empty:
            print(f"Skip {file}: no valid data after cleaning")
            file_info.append({
                'filename': file,
                'status': 'skipped',
                'reason': 'no valid data after cleaning',
                'shape': original_shape,
                'clean_shape': (0, 0)
            })
            continue
        
        file_names.append(file)
        for feat in FEATURES:
            col_data[feat].append(df_clean[feat].values)
        
        # 记录文件信息
        file_info.append({
            'filename': file,
            'status': 'loaded',
            'original_shape': original_shape,
            'clean_shape': df_clean.shape,
            'features_stats': {feat: {
                'mean': df_clean[feat].mean(),
                'std': df_clean[feat].std(),
                'min': df_clean[feat].min(),
                'max': df_clean[feat].max(),
                'null_count': df_clean[feat].isnull().sum()
            } for feat in FEATURES}
        })
        
        print(f"Loaded {file}: {original_shape} -> {df_clean.shape}")
        
    except Exception as e:
        print(f"Error loading {file}: {e}")
        file_info.append({
            'filename': file,
            'status': 'error',
            'reason': str(e)
        })

print(f"Successfully loaded {len(file_names)} samples out of {len(selected_files)} specified files.")

# ------------------------------------------------------------------
# 3. Interactive file selection (optional)
# ------------------------------------------------------------------
def interactive_file_selection():
    """
    提供交互式文件选择功能
    """
    available_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    
    print("\n" + "="*60)
    print("INTERACTIVE FILE SELECTION")
    print("="*60)
    print("Available CSV files:")
    for i, fname in enumerate(available_files, 1):
        print(f"  {i:3d}. {fname}")
    
    print("\nEnter file numbers separated by spaces (e.g., 1 5 10 15):")
    print("Or enter 'all' to use all files:")
    print("Or press Enter to use current selection:")
    
    try:
        user_input = input("Your choice: ").strip()
        
        if user_input.lower() == 'all':
            return available_files
        elif user_input == '':
            return selected_files
        else:
            indices = [int(x) - 1 for x in user_input.split()]
            selected = [available_files[i] for i in indices if 0 <= i < len(available_files)]
            return selected
    except (ValueError, IndexError):
        print("Invalid input. Using current selection.")
        return selected_files

# 如果需要交互式选择，取消下面的注释
# selected_files = interactive_file_selection()

# 继续使用原有的绘图代码...
# ------------------------------------------------------------------
# 4. Plot overlay curves with improved layout
# ------------------------------------------------------------------
if len(file_names) > 0:
    # 重新组织特征：传感器1 vs 传感器2
    sensor1_features = ["x1", "y1", "z1"]
    sensor2_features = ["x2", "y2", "z2"]
    axis_labels = ["X-axis", "Y-axis", "Z-axis"]
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 定义颜色列表
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_names)))
    
    # 绘制传感器1的数据（左列）
    for row_idx, (feat1, feat2, axis_name) in enumerate(zip(sensor1_features, sensor2_features, axis_labels)):
        # 左列：传感器1
        ax1 = axes[row_idx, 0]
        for i, (series, fname) in enumerate(zip(col_data[feat1], file_names)):
            ax1.plot(series, alpha=0.7, color=colors[i], linewidth=1.5,
                    label=f"{fname[:15]}..." if len(fname) > 15 else fname)
        
        ax1.set_title(f"Sensor 1 - {axis_name} ({feat1})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time index", fontsize=12)
        ax1.set_ylabel("Amplitude", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        all_values1 = np.concatenate(col_data[feat1])
        ax1.text(0.02, 0.98, f"Range: [{np.min(all_values1):.0f}, {np.max(all_values1):.0f}]\n"
                              f"Mean: {np.mean(all_values1):.1f}", 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 fontsize=10)
        
        # 右列：传感器2
        ax2 = axes[row_idx, 1]
        for i, (series, fname) in enumerate(zip(col_data[feat2], file_names)):
            ax2.plot(series, alpha=0.7, color=colors[i], linewidth=1.5,
                    label=f"{fname[:15]}..." if len(fname) > 15 else fname)
        
        ax2.set_title(f"Sensor 2 - {axis_name} ({feat2})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Time index", fontsize=12)
        ax2.set_ylabel("Amplitude", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        all_values2 = np.concatenate(col_data[feat2])
        ax2.text(0.02, 0.98, f"Range: [{np.min(all_values2):.0f}, {np.max(all_values2):.0f}]\n"
                              f"Mean: {np.mean(all_values2):.1f}", 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 fontsize=10)
    
    # 在第一行添加图例
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 添加列标题
    fig.text(0.25, 0.95, 'Magnetometer Sensor 1', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.75, 0.95, 'Magnetometer Sensor 2', ha='center', fontsize=16, fontweight='bold')
    
    plt.suptitle(f"Dual-Sensor Magnetic Signature Analysis ({len(file_names)} specified samples)", 
                fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 保存图片
    plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {FIG_PATH}")
    plt.show()
    
else:
    print("No valid data loaded. Cannot generate plots.")


