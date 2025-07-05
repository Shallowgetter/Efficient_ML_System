"""
eTraM Static 数据集结构快速可视化 / 统计脚本
"""

import os, json, math, h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------- 0  全局配置 ----------
CONFIG = {
    "root": Path("/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/eTraM/Static"),
    "out_dir": Path("analysis_output"),
    "sample_seconds": 30,      
    "heatmap_size": (720, 1280),
    "polarity_scale": 3,       
}

palette = {0:'yellow',1:'cyan',2:'lime',3:'orange',
           4:'magenta',5:'blue',6:'red',7:'white'}

CONFIG["out_dir"].mkdir(exist_ok=True)

# ---------- 1  数据索引 ----------
def scan_dataset(root):
    rows = []
    for split_dir in ["HDF5", "RAW"]:
        for sub in (root / split_dir).rglob("*_td.h5"):
            split = "train" if "train" in sub.parts else \
                    "val"   if "val"   in sub.parts else "test"
            base = sub.with_suffix("").as_posix()[:-3]
            bbox = Path(base + "_bbox.npy")
            rows.append(dict(split=split, h5=sub, bbox=bbox))
    df = pd.DataFrame(rows).sort_values(["split", "h5"])
    df.to_csv(CONFIG["out_dir"]/ "dataset_index.csv", index=False)
    return df

# ---------- 2  事件可视化 ----------
def quick_event_view(h5_path, seconds, out_png):
    with h5py.File(h5_path,'r') as f:
        # 先检查文件结构
        print(f"检查文件结构: {h5_path.name}")
        print(f"根级键: {list(f.keys())}")
        
        # 处理 events 组结构
        if "events" in f:
            evs = f["events"]
            print(f"events 结构: {type(evs)}")
            if hasattr(evs, 'keys'):
                print(f"events 键: {list(evs.keys())}")
                
                # 读取各个字段
                ts = evs["t"][:]
                xs = evs["x"][:]
                ys = evs["y"][:]
                ps = evs["p"][:]
                
                # 时间筛选
                t0 = ts[0]
                mask = ts < t0 + seconds * 1e6
                
                # 应用掩码
                ts = ts[mask]
                xs = xs[mask]
                ys = ys[mask]
                ps = ps[mask]
                
            elif hasattr(evs, 'dtype') and evs.dtype.names:
                # 结构化数组格式
                t0 = evs["t"][0]
                mask = evs["t"][:] < t0 + seconds*1e6
                xs, ys, ps = evs["x"][mask], evs["y"][mask], evs["p"][mask]
                ts = evs["t"][mask]
            else:
                # 普通数组格式，假设列顺序为 [t, x, y, p]
                data = evs[:]
                t0 = data[0, 0]
                mask = data[:, 0] < t0 + seconds*1e6
                selected_data = data[mask]
                ts, xs, ys, ps = selected_data[:, 0], selected_data[:, 1], selected_data[:, 2], selected_data[:, 3]
        else:
            # 尝试其他可能的键名
            possible_keys = ["data", "event_data", "dvs_events"]
            evs = None
            for key in possible_keys:
                if key in f:
                    evs = f[key]
                    break
            
            if evs is None:
                raise KeyError(f"未找到事件数据，可用键: {list(f.keys())}")
            
            # 处理找到的数据
            data = evs[:]
            t0 = data[0, 0]
            mask = data[:, 0] < t0 + seconds*1e6
            selected_data = data[mask]
            ts, xs, ys, ps = selected_data[:, 0], selected_data[:, 1], selected_data[:, 2], selected_data[:, 3]

    print(f"事件数量: {len(ts)}, 时间范围: {(ts[-1] - ts[0])/1e6:.2f}s")
    
    H, W = CONFIG["heatmap_size"]
    img_r = np.zeros((H, W), np.int32)
    img_b = np.zeros_like(img_r)
    
    for x, y, p in zip(xs, ys, ps):
        x, y = int(x), int(y)
        if 0 <= x < W and 0 <= y < H:  # 边界检查
            if p > 0: 
                img_r[y, x] += 1
            else:   
                img_b[y, x] += 1
    
    vis = np.stack([np.log1p(img_r) * CONFIG["polarity_scale"],
                    np.zeros_like(img_r),
                    np.log1p(img_b) * CONFIG["polarity_scale"]], axis=-1)
    
    if vis.max() > 0:
        vis = (vis / vis.max() * 255).astype(np.uint8)
    else:
        vis = vis.astype(np.uint8)

    # 时间直方图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(vis)
    ax1.set_title(f"{h5_path.name}  (+/- polarity)")
    
    # time hist
    bins = 100
    ts_rel = (ts - ts[0]) / 1e6  # 使用实际的t0
    ax2.hist(ts_rel, bins=bins)
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("#events")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------- 3  BBox 可视化 ----------
def bbox_overlay_view(h5_path, bbox_path, seconds, out_png):
    quick_event_view(h5_path, seconds, out_png)  # 先生成底图
    bbox = np.load(bbox_path)
    
    # 检查bbox数组的结构
    print(f"BBox shape: {bbox.shape}")
    print(f"BBox dtype: {bbox.dtype}")
    if len(bbox) > 0:
        print(f"First few entries: {bbox[:3]}")
    
    # 根据实际结构处理bbox数据
    if bbox.ndim == 1:
        # 如果是一维数组，可能是结构化数组
        if bbox.dtype.names:
            # 结构化数组
            t0 = bbox["t"][0] if "t" in bbox.dtype.names else bbox[0]
            if "t" in bbox.dtype.names:
                mask = bbox["t"] < t0 + seconds*1e6
                sel = bbox[mask]
            else:
                # 假设第一个字段是时间
                mask = bbox[bbox.dtype.names[0]] < t0 + seconds*1e6
                sel = bbox[mask]
        else:
            # 普通一维数组，可能需要reshape
            print("Warning: 1D array without field names, skipping bbox overlay")
            return
    else:
        # 二维数组
        t0 = bbox[0,0]
        mask = bbox[:,0] < t0 + seconds*1e6
        sel = bbox[mask]
    
    img = plt.imread(out_png)
    fig,ax = plt.subplots(figsize=(6,4))
    ax.imshow(img)
    ax.axis("off")
    
    # 根据数据结构绘制bbox
    if bbox.dtype.names:
        # 结构化数组 - 直接通过字段名访问
        for item in sel:
            # 获取坐标和尺寸
            x = item["x"]
            y = item["y"] 
            w = item["w"]
            h = item["h"]
            
            # 获取类别ID
            if "class_id" in bbox.dtype.names:
                c = item["class_id"]
            elif "c" in bbox.dtype.names:
                c = item["c"]
            elif "class" in bbox.dtype.names:
                c = item["class"]
            else:
                c = 0
            
            rect = plt.Rectangle((x,y), w, h, fill=False,
                               edgecolor=palette.get(int(c), "white"), linewidth=1)
            ax.add_patch(rect)
    else:
        # 普通数组，假设格式为 [t,x,y,w,h,c,oid,conf]
        for row in sel:
            if len(row) >= 6:
                t,x,y,w,h,c = row[:6]
                rect = plt.Rectangle((x,y), w, h, fill=False,
                                   edgecolor=palette.get(int(c), "white"), linewidth=1)
                ax.add_patch(rect)
    
    fig.tight_layout()
    # 正确处理Path对象生成bbox输出文件名
    bbox_out_png = out_png.parent / (out_png.stem + "_bbox.png")
    fig.savefig(bbox_out_png)
    plt.close(fig)

# ---------- 4  类别/密度统计 ----------
def class_stats(df):
    records=[]
    for _,row in tqdm(df.iterrows(), total=len(df), desc="Stats"):
        arr = np.load(row.bbox)
        
        # 检查数组结构
        if arr.dtype.names:
            # 结构化数组
            for item in arr:
                # 获取类别ID
                if "class_id" in arr.dtype.names:
                    cls = int(item["class_id"])
                elif "c" in arr.dtype.names:
                    cls = int(item["c"])
                elif "class" in arr.dtype.names:
                    cls = int(item["class"])
                else:
                    cls = 0
                
                # 获取尺寸
                w = item["w"]
                h = item["h"]
                
                # 获取track_id
                if "track_id" in arr.dtype.names:
                    oid = int(item["track_id"])
                elif "oid" in arr.dtype.names:
                    oid = int(item["oid"])
                else:
                    oid = 0
                
                records.append(dict(
                    split=row.split, 
                    cls=cls,
                    area=float(w) * float(h),
                    oid=oid
                ))
        else:
            # 普通数组
            if arr.ndim == 2 and arr.shape[1] >= 6:
                for t,x,y,w,h,c,*rest in arr:
                    oid = rest[0] if len(rest) > 0 else 0
                    records.append(dict(split=row.split, cls=int(c),
                                       area=w*h, oid=int(oid)))
            else:
                print(f"Warning: Unexpected bbox format in {row.bbox}")
                continue
                
    if records:
        stat = pd.DataFrame(records)
        res = (stat.groupby(["split","cls"])
                     .agg(bbox_cnt=("cls","size"),
                          obj_cnt=("oid","nunique"),
                          area_mean=("area","mean"))
                     .reset_index())
        print(res)
        res.to_json(CONFIG["out_dir"]/ "class_stats.json", orient="records", indent=2)
    else:
        print("No valid bbox records found")

def temporal_density_stats(df):
    ev_per_sec=[]
    for _,row in tqdm(df.sample(20).iterrows(), total=20, desc="Density"):
        with h5py.File(row.h5,'r') as f:
            ts = f["events"]["t"][:]
        dur = (ts[-1]-ts[0])/1e6
        ev_per_sec.append(len(ts)/dur)
    plt.boxplot(ev_per_sec, vert=False)
    plt.xlabel("#events / sec"); plt.title("Sparsity overview")
    plt.savefig(CONFIG["out_dir"]/ "density_boxplot.png"); plt.close()

# ---------- 5  主流程 ----------
if __name__=="__main__":
    df = scan_dataset(CONFIG["root"])
    for _,row in df.sample(5).iterrows():  # 抽 5 个文件做示例
        out = CONFIG["out_dir"]/ f"{row.h5.stem}.png"
        quick_event_view(row.h5, CONFIG["sample_seconds"], out)
        if row.bbox.exists():
            bbox_overlay_view(row.h5, row.bbox,
                              CONFIG["sample_seconds"], out)
    class_stats(df)
    temporal_density_stats(df)
    print("分析完成！查看 analysis_output 目录.")
