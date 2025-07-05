"""
bboxSeqAnalysis.py
==================
可视化 eTraM Static 数据中单个 object_id 的 (x, y, w, h) 随时间变化曲线。
Author : HPC Efficient_Computing_System 组
Date   : 2025-07-03
依赖   : numpy, matplotlib, pandas, h5py, tqdm
运行   : python bboxSeqAnalysis.py
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ──────────────────────────────
# 0.  全局配置
# ──────────────────────────────
CFG = {
    "root": Path("/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/eTraM"),        # 数据根目录
    "out":  Path("bbox_seq"),            # 输出目录
    "min_len": 20,                       # 轨迹最小帧数
    "pick": "longest",                   # "longest" | "random"
    "num_random": 3,                     # 随机采样条数
}
CFG["out"].mkdir(exist_ok=True, parents=True)


# ──────────────────────────────
# 1.  扫描全部 bbox 文件
# ──────────────────────────────
def list_bbox_files(root: Path):
    return list(root.rglob("*_bbox.npy"))


# ──────────────────────────────
# 2.  解析并分组轨迹
# ──────────────────────────────
def load_tracks(bbox_path: Path):
    """
    返回轨迹 dict:
        {obj_id: {"cls": class_id, "t": ..., "x": ..., "y": ..., "w": ..., "h": ...}}
    时间单位 µs → s
    """
    try:
        arr = np.load(bbox_path)
        
        # 处理一维数组的情况
        if arr.ndim == 1:
            # 检查是否能被7整除（每行7列数据）
            if len(arr) % 7 == 0:
                print(f"提示: {bbox_path} 是一维数组，尝试重新整形为 ({len(arr)//7}, 7)")
                arr = arr.reshape(-1, 7)
            else:
                print(f"警告: {bbox_path} 一维数组长度 {len(arr)} 无法整除7，跳过")
                return {}
        
        # 检查数组维度和形状
        if arr.ndim != 2:
            print(f"警告: {bbox_path} 数组维度异常 (维度: {arr.ndim}, 形状: {arr.shape})")
            return {}
        
        if arr.shape[1] < 7:
            print(f"警告: {bbox_path} 列数不足 (期望7列，实际{arr.shape[1]}列)")
            return {}
        
        if arr.shape[0] == 0:
            print(f"警告: {bbox_path} 数据为空")
            return {}
        
        t, x, y, w, h, cid, oid = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5].astype(int), arr[:, 6].astype(int)
        tracks = {}
        for oid_ in np.unique(oid):
            mask = oid == oid_
            if mask.sum() < CFG["min_len"]:
                continue
            tracks[oid_] = dict(
                cls=int(cid[mask][0]),
                t=(t[mask] - t[mask][0]) / 1e6,   # 秒
                x=x[mask] + w[mask] / 2,          # 中心点
                y=y[mask] + h[mask] / 2,
                w=w[mask], h=h[mask]
            )
        return tracks
        
    except Exception as e:
        print(f"错误: 处理 {bbox_path} 时出现异常: {e}")
        return {}


# ──────────────────────────────
# 3.  绘图
# ──────────────────────────────
def plot_track(track, save_path: Path, title_suffix=""):
    """
    track: dict(t,x,y,w,h,cls)
    """
    t = track["t"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    # x_center
    axes[0, 0].plot(t, track["x"])
    axes[0, 0].set_ylabel("x_center")
    # y_center
    axes[1, 0].plot(t, track["y"])
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("y_center")
    # width
    axes[0, 1].plot(t, track["w"])
    axes[0, 1].set_ylabel("width")
    # height
    axes[1, 1].plot(t, track["h"])
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("height")

    # 嵌入空间轨迹 (右上角占位)
    ax_in = fig.add_axes([0.65, 0.6, 0.3, 0.3])
    ax_in.plot(track["x"], track["y"], marker="o", linewidth=1)
    ax_in.set_title("Spatial path", fontsize=8)
    ax_in.set_xlabel("x"); ax_in.set_ylabel("y")
    ax_in.invert_yaxis()     # y 轴朝下符合图像坐标
    ax_in.tick_params(labelsize=6)

    fig.suptitle(f"Obj {title_suffix} | cls={track['cls']}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ──────────────────────────────
# 4.  主流程
# ──────────────────────────────
def main():
    bbox_files = list_bbox_files(CFG["root"])
    print(f"发现 bbox 文件 {len(bbox_files)} 个")
    
    # 检查前几个文件的数据结构（可选）
    print("检查前3个文件的数据结构:")
    for bbox_path in bbox_files[:3]:
        inspect_bbox_file(bbox_path)
    
    processed_count = 0
    error_count = 0
    
    for bbox_path in tqdm(bbox_files, desc="Processing"):
        tracks = load_tracks(bbox_path)
        if not tracks:
            error_count += 1
            continue

        if CFG["pick"] == "longest":
            # 轨迹长度最长
            oid = max(tracks.keys(), key=lambda k: len(tracks[k]["t"]))
            track_list = [(oid, tracks[oid])]
        else:  # random
            oids = random.sample(list(tracks.keys()), min(CFG["num_random"], len(tracks)))
            track_list = [(oid, tracks[oid]) for oid in oids]

        # 绘图
        for oid, tr in track_list:
            out_png = CFG["out"] / f"{bbox_path.stem}_oid{oid}.png"
            plot_track(tr, out_png, title_suffix=f"{oid} ({bbox_path.stem})")
        
        processed_count += 1

    print(f"全部轨迹已保存至 {CFG['out'].resolve()}")
    print(f"成功处理: {processed_count} 个文件，跳过/错误: {error_count} 个文件")

def inspect_bbox_file(bbox_path: Path):
    """检查bbox文件的数据结构"""
    try:
        arr = np.load(bbox_path)
        print(f"文件: {bbox_path.name}")
        print(f"  形状: {arr.shape}")
        print(f"  数据类型: {arr.dtype}")
        print(f"  前10个值: {arr.flat[:10]}")
        print(f"  能否重新整形为7列: {len(arr) % 7 == 0 if arr.ndim == 1 else 'N/A'}")
        print("-" * 50)
    except Exception as e:
        print(f"检查文件 {bbox_path} 时出错: {e}")

if __name__ == "__main__":
    main()
