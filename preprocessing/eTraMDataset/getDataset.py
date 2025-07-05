"""
Dataset preparation for eTraM sequence data.
"""

from pathlib import Path
from functools import lru_cache

import yaml, h5py, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2


def load_cfg(yaml_path="etram_seq_params.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_cfg()

def _voxelize(events, t0, t1, bins, H, W):
    """
    events: ndarray (n,4) [t,x,y,p]  t:μs
    return: tensor (2*bins, H, W)
    """
    C = 2 * bins
    voxel = np.zeros((C, H, W), dtype=np.float32)
    if len(events) == 0:
        return voxel

    dt = (t1 - t0) / bins
    bin_idx = np.clip(((events[:, 0] - t0) // dt).astype(int), 0, bins - 1)
    xs = np.clip(events[:, 1], 0, W - 1)
    ys = np.clip(events[:, 2], 0, H - 1)
    pol = (events[:, 3] > 0).astype(int)  # 1:正,0:负

    for b, x, y, p in zip(bin_idx, xs, ys, pol):
        voxel[p * bins + b, y, x] += 1.0

    # log-normalize
    voxel = np.log1p(voxel)
    return voxel


def _crop_and_resize(events, bbox, resize, expand):
    """
    events: ndarray (n,4)
    bbox  : (min_x,min_y,max_x,max_y)
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0 - expand))
    y0 = max(0, int(y0 - expand))
    x1 = int(x1 + expand)
    y1 = int(y1 + expand)

    # 裁剪
    m = (events[:, 1] >= x0) & (events[:, 1] < x1) & \
        (events[:, 2] >= y0) & (events[:, 2] < y1)
    ev = events[m].copy()
    ev[:, 1] -= x0
    ev[:, 2] -= y0

    # 缩放
    h, w = y1 - y0, x1 - x0
    scale_x = resize / w
    scale_y = resize / h
    ev[:, 1] = (ev[:, 1] * scale_x).astype(np.int32)
    ev[:, 2] = (ev[:, 2] * scale_y).astype(np.int32)
    ev[:, 1] = np.clip(ev[:, 1], 0, resize - 1)
    ev[:, 2] = np.clip(ev[:, 2], 0, resize - 1)

    return ev, (resize, resize)


def _bbox_union(track, t_s, t_e):
    """
    从轨迹中取 [t_s, t_e] 内所有 bbox，返回最大覆盖框
    """
    m = (track[:, 0] >= t_s) & (track[:, 0] <= t_e)
    if not m.any():
        return None
    xs = track[m, 1]
    ys = track[m, 2]
    ws = track[m, 3]
    hs = track[m, 4]
    x0, y0 = xs.min(), ys.min()
    x1 = (xs + ws).max()
    y1 = (ys + hs).max()
    return x0, y0, x1, y1


@lru_cache(maxsize=None)
def _open_h5(h5_path):
    return h5py.File(h5_path, "r")


# ---------- 3  数据集 ----------

class ETramSeqDataset(Dataset):
    """
    每个样本：
        x : torch.FloatTensor  (seq_len, 2*time_bins, H, W)
        y : int  (class_id)
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.window = int(cfg["dataset"]["window_sec"] * 1e6)   # μs
        self.stride = int(cfg["dataset"]["stride_sec"] * 1e6)   # μs
        self.seq_len = cfg["dataset"]["seq_len"]
        self.bins = cfg["dataset"]["time_bins"]
        self.resize = cfg["dataset"]["resize"]
        self.expand = cfg["dataset"]["crop_expand"]
        self.min_track = cfg["dataset"]["min_track_frames"]

        self.samples = self._build_index()

    def _build_index(self):
        root = Path(self.cfg["dataset"]["root"])
        split = self.mode
        entries = []
        for h5_path in root.rglob(f"*{split}*_td.h5"):
            bbox_path = h5_path.with_name(h5_path.name.replace("_td.h5", "_bbox.npy"))
            if not bbox_path.exists():
                continue
            bboxes = np.load(bbox_path)
            # 按 object_id 分组
            oid_list = np.unique(bboxes[:, 6]).astype(int)
            for oid in oid_list:
                track = bboxes[bboxes[:, 6] == oid]
                if len(track) < self.min_track:
                    continue
                t_start, t_end = track[0, 0], track[-1, 0]
                cur = t_start
                while cur + self.window * self.seq_len <= t_end:
                    entries.append((h5_path, bbox_path, oid, cur))
                    cur += self.stride
        return entries

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, bbox_path, oid, t0 = self.samples[idx]
        track = np.load(bbox_path)
        track = track[track[:, 6] == oid]

        seq = []
        for i in range(self.seq_len):
            win_s = t0 + i * self.window
            win_e = win_s + self.window

            # 1) 取窗内事件
            f = _open_h5(str(h5_path))
            events_grp = f["events"]
            # 二分索引加速
            l = np.searchsorted(events_grp["t"], win_s, side="left")
            r = np.searchsorted(events_grp["t"], win_e, side="right")
            ev = np.stack([events_grp["t"][l:r],
                           events_grp["x"][l:r],
                           events_grp["y"][l:r],
                           events_grp["p"][l:r]], axis=1)

            # 2) 动态 bbox 裁剪
            bbox = _bbox_union(track, win_s, win_e)
            if bbox is None:
                # 无 bbox，返回全图
                bbox = (0, 0, events_grp["x"].attrs["resolution"][0],
                              events_grp["y"].attrs["resolution"][1])
            ev_crop, (H, W) = _crop_and_resize(ev, bbox, self.resize, self.expand)

            # 3) 体素化
            voxel = _voxelize(ev_crop, win_s, win_e, self.bins, H, W)
            seq.append(voxel)

        x = torch.from_numpy(np.stack(seq))              # (seq_len, C, H, W)
        y = int(track[0, 5])                             # class_id
        return x, y


# ---------- 4  DataLoader 帮助函数 ----------
def build_dataloader(mode="train"):
    ds = ETramSeqDataset(CFG, mode)
    dl = DataLoader(ds,
                    batch_size=CFG["dataloader"]["batch_size"],
                    shuffle=(mode == "train"),
                    num_workers=CFG["dataloader"]["num_workers"],
                    pin_memory=True)
    return dl


# ---------- 5  简短 sanity check ----------
if __name__ == "__main__":
    dl = build_dataloader("train")
    for batch_idx, (x, y) in enumerate(tqdm(dl, desc="Loading")):
        print(f"x {x.shape}  y {y.shape}  {y.unique()}")
        if batch_idx == 3:          # 只看几批
            break
