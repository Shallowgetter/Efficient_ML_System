"""
Reproduce XGBoost baseline for the SHL locomotion dataset using the custom
`getDataset.py` preprocessing pipeline shipped with the Efficient_Computing_System
project.

"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils.utils import plot_confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoost not found – install via `pip install xgboost>=2.0`.") from e

# -----------------------------------------------------------------------------
# 1. Configuration – amend these paths & window params to your environment
# -----------------------------------------------------------------------------
from preprocessing.getDataset import SHLConfig, process_and_save_dataset  # noqa: E402

CFG_RAW_ROOT = Path("/Users/xiangyifei/Documents/HPC_Efficient_Computing_System/dataset/SHL")               # ← EDIT ME
OUTPUT_DIR = Path("localExperiments/model_result/XGBoost_result")                      # artefacts
WINDOW_SIZE = 500                                            # 5 s @ 100 Hz
OVERLAP = 0.0                                               # 0 % window overlap
TRAIN_RATIO = 0.7                                          # 87 / 30 split when
                                                             # constructing NPZ
SEED = 42

# Class‑name list (edit to match SHL label order if需要)
CLASS_NAMES: List[str] = [f"Class_{i}" for i in range(8)]

# Adjust GPU usage automatically
TREE_METHOD = "gpu_hist" if os.getenv("CUDA_VISIBLE_DEVICES", "") else "hist"

# -----------------------------------------------------------------------------
# 2. Helper – statistical feature extraction
# -----------------------------------------------------------------------------

STAT_FUNCS = {
    "mean":   np.mean,
    "std":    np.std,
    "min":    np.min,
    "max":    np.max,
    "p25":    lambda x, axis: np.percentile(x, 25, axis=axis),
    "p75":    lambda x, axis: np.percentile(x, 75, axis=axis),
}


def _extract_features(x_3d: np.ndarray) -> np.ndarray:
    """Vectorise *N×F×W* sensor windows into *N×(F·K)* feature matrix."""
    N, F, _ = x_3d.shape
    feats: List[np.ndarray] = []
    for fn in STAT_FUNCS.values():
        feats.append(fn(x_3d, axis=2))  # -> (N, F)
    return np.hstack(feats).astype(np.float32)  # (N, F·K)


# -----------------------------------------------------------------------------
# 3. Dataset preparation (runs once) – skip if NPZ already exists
# -----------------------------------------------------------------------------

def maybe_prepare_npz():
    # 检查多个可能的位置
    data_dir = Path("data")  # 你提到的data文件夹
    out_npz_train_data = data_dir / "shl_train.npz"
    out_npz_val_data = data_dir / "shl_validation.npz"
    
    out_npz_train = OUTPUT_DIR / "shl_train.npz"
    out_npz_val = OUTPUT_DIR / "shl_validation.npz"

    # 优先检查data文件夹中的NPZ文件
    if out_npz_train_data.exists() and out_npz_val_data.exists():
        print("✔ Found existing NPZ files in data folder – using those.")
        # 如果OUTPUT_DIR中没有文件，创建软链接或复制文件
        if not (out_npz_train.exists() and out_npz_val.exists()):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(out_npz_train_data, out_npz_train)
            shutil.copy2(out_npz_val_data, out_npz_val)
            print(f"✔ Copied NPZ files from data folder to {OUTPUT_DIR}")
        return
    
    # 检查OUTPUT_DIR中的文件
    if out_npz_train.exists() and out_npz_val.exists():
        print("✔ Found existing NPZ files in output directory – skipping raw preprocessing.")
        return

    # 如果都不存在，才进行处理
    cfg = SHLConfig(
        root=CFG_RAW_ROOT,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
    )
    print("⏳ Processing raw SHL recordings …")
    process_and_save_dataset(cfg, OUTPUT_DIR, train_ratio=TRAIN_RATIO, seed=SEED)
    print("✔ NPZ files written to", OUTPUT_DIR)


# -----------------------------------------------------------------------------
# 4. Main training routine
# -----------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    maybe_prepare_npz()

    # Load pre‑chunked windows
    train_npz = np.load(OUTPUT_DIR / "shl_train.npz")
    X_train_raw: np.ndarray = train_npz["data"]  # (N, F, W)
    y_train: np.ndarray = train_npz["labels"]      # (N,)

    val_npz = np.load(OUTPUT_DIR / "shl_validation.npz")
    X_val_raw: np.ndarray = val_npz["data"]
    y_val: np.ndarray = val_npz["labels"]

    # Feature engineering (vectorise)
    print("Extracting statistical features …")
    X_train = _extract_features(X_train_raw)
    X_val = _extract_features(X_val_raw)

    print(f"Feature matrix shapes | train: {X_train.shape}  val: {X_val.shape}")

    # 3‑Fold CV with stratification on the *window* labels
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # Hyper‑parameters ported & slightly tuned
    xgb_params: Dict = {
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_train)),
        "max_depth": 21,
        "min_child_weight": 10,
        "learning_rate": 0.0102,
        "gamma": 0.005,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "n_estimators": 1000,
        "tree_method": TREE_METHOD,
        "eval_metric": "merror",
        "random_state": SEED,
    }

    oof_pred = np.zeros((len(X_train), xgb_params["num_class"]))
    val_pred_agg = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"\n★ Fold {fold} – train {len(train_idx)} | valid {len(valid_idx)}")
        
        # 在XGBoost 3.x中，早停参数需要在初始化时设置
        xgb_params_with_early_stop = xgb_params.copy()
        xgb_params_with_early_stop.update({
            "early_stopping_rounds": 65,
            "enable_categorical": False,  # 明确禁用分类特征
        })
        
        clf = XGBClassifier(**xgb_params_with_early_stop)

        clf.fit(
            X_train[train_idx], y_train[train_idx],
            eval_set=[(X_train[train_idx], y_train[train_idx]),
                      (X_train[valid_idx], y_train[valid_idx])],
            verbose=200,
        )

        # Out‑of‑fold
        oof_pred[valid_idx] = clf.predict_proba(X_train[valid_idx])
        # External validation
        val_pred_agg.append(clf.predict_proba(X_val))

        # Persist model
        joblib.dump(clf, OUTPUT_DIR / f"model_fold{fold}.joblib", compress=3)

    # ------------------------------------------------------------------
    # 5. Metrics & artefact persistence
    # ------------------------------------------------------------------
    oof_labels = oof_pred.argmax(axis=1)
    acc = accuracy_score(y_train, oof_labels)
    f1 = f1_score(y_train, oof_labels, average="macro")
    print(f"\nOverall OOF accuracy: {acc:.4f} | macro‑F1: {f1:.4f}")

    # Average the validation predictions across folds (soft voting)
    val_pred_mean = np.mean(val_pred_agg, axis=0)

    # Save probability matrices
    np.save(OUTPUT_DIR / "oof_pred.npy", oof_pred)
    np.save(OUTPUT_DIR / "val_pred.npy", val_pred_mean)

    # ------------------------------------------------------------------
    # 6. Confusion‑matrix visualisation on external validation set
    # ------------------------------------------------------------------
    val_pred_labels = val_pred_mean.argmax(axis=1)
    print("Generating confusion matrix …")
    plot_confusion_matrix(
        y_val,
        val_pred_labels,
        class_names=CLASS_NAMES,
        normalize=True,
        fontsize=16,
        vmin=0,
        vmax=1,
        axis=1,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_val.png", dpi=150)
    plt.close()
    print("✔ Confusion matrix saved to", OUTPUT_DIR / "confusion_matrix_val.png")

    # Optional: JSON summary
    summary = {
        "oof_accuracy": acc,
        "oof_macroF1": f1,
        "folds": skf.get_n_splits(),
        "features_per_window": X_train.shape[1],
        "tree_method": TREE_METHOD,
    }
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print("\n✔ All artefacts saved to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
