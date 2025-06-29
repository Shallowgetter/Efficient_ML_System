"""
用法（示例）:
$ python shl_data_pipeline.py \
      --raw_dir  /path/to/SHL/raw/Participant1/ \
      --out_dir  /path/to/SHL/preprocessed/ \
      --positions hand torso bag hips \
      --rot_quat_col 4 5 6 7          # 若文件第 4–7 列为四元数
"""

import argparse, json, pickle, re, sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

ROOT_PATH = '/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/SHL_2020'
OUT_PATH = 'data/SHL_2020'


# ---------- I/O ----------
def read_txt(path: Path, usecols: Tuple[int, ...]) -> np.ndarray:
    """按列次序载入纯文本传感器文件；支持缺失值过滤。"""
    data = np.loadtxt(path, usecols=usecols, dtype=np.float32)
    return data[~np.isnan(data).any(axis=1)]


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------- 数学工具 ----------
def quaternion_rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    将 Nx3 加速度向量转换到体坐标系。
    vec : (N,3); quat : (N,4)  [x,y,z,w] ➜ 返回同形矩阵
    """
    rot = R.from_quat(quat.astype(np.float64))
    return rot.apply(vec)


def compute_jerk(acc: np.ndarray, fs: int = 100) -> np.ndarray:
    """Jerk = d𝐚/dt；一阶差分后乘采样频率 fs。"""
    jerk = np.vstack([np.zeros((1, acc.shape[1])), np.diff(acc, axis=0)]) * fs
    return jerk


def split_vertical_horizontal(acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """参考原仓库，将重力方向定义为 z 轴后，返回水平/垂直分量模长。"""
    norm = np.linalg.norm(acc, axis=1, keepdims=True) + 1e-8
    vertical = np.abs(acc[:, 2:3])                      # 竖直 = z 分量绝对值
    horizontal = np.sqrt((acc[:, :2] ** 2).sum(axis=1, keepdims=True))
    return horizontal, vertical, norm


# ---------- 主流程 ----------
def process_single_txt(txt_path: Path,
                       quat_cols: Tuple[int, ...] | None,
                       fs: int = 100,
                       rotate: bool = True) -> pd.DataFrame:
    """
    读取单个文件并生成完整通道：
    ACC (x,y,z,|a|), LACC, ACCH, ACCV, JERK, JERKH, JERKV
    """
    # 假设文件列顺序: [t, ax, ay, az, gx, gy, gz, qx, qy, qz, qw]
    arr = read_txt(txt_path, tuple(range(1, 11)))
    acc_raw, quat = arr[:, :3], arr[:, 7:] if quat_cols else (arr[:, :3], None)

    if rotate and quat_cols:
        acc_raw = quaternion_rotate(acc_raw, quat)

    jerk = compute_jerk(acc_raw, fs)
    h_acc, v_acc, acc_norm = split_vertical_horizontal(acc_raw)
    h_jerk, v_jerk, _ = split_vertical_horizontal(jerk)

    df = pd.DataFrame(
        np.hstack([acc_raw, acc_norm, h_acc, v_acc,
                   jerk, h_jerk, v_jerk]),
        columns=[
            "ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG",
            "ACCH", "ACCV",
            "JERK_X", "JERK_Y", "JERK_Z",
            "JERKH", "JERKV",
        ],
    )
    df["TS"] = arr[:, 0]  # 原时间戳保留以便窗口分割
    return df


def batch_convert(raw_dir: Path,
                  out_dir: Path,
                  positions: Tuple[str, ...],
                  **kwargs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_files = [p for p in raw_dir.rglob("*.txt") if any(pos in p.stem for pos in positions)]
    if not txt_files:
        sys.exit(f"[!] 未找到符合位置 {positions} 的 .txt 文件")

    for txt in tqdm(txt_files, desc="Preprocessing"):
        df = process_single_txt(txt, **kwargs)
        save_path = out_dir / f"{txt.stem}.pkl"
        save_pickle(df, save_path)


# ---------- CLI ----------
def main():
    # 直接使用预定义的路径，不使用命令行参数
    raw_dir = Path(ROOT_PATH)
    out_dir = Path(OUT_PATH)
    positions = ["hand", "torso", "bag", "hips"]
    quat_cols = (7, 8, 9, 10)  # 假设四元数在第7-10列，根据实际情况调整
    
    batch_convert(
        raw_dir,
        out_dir,
        tuple(positions),
        quat_cols=quat_cols,
    )
    print(f"[✓] 处理完成，结果已保存在 {out_dir}")


if __name__ == "__main__":
    main()
