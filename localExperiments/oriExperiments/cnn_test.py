# ori_cnn_experiment.py
# -*- coding: utf-8 -*-
"""
Pure 2-D CNN baseline for ORI sequence dataset (4-class)
Input shape : (B, 10, 1008)  →  treated as (B, 1, 10, 1008)
"""
import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------- #
# 0) 路径与工具
# ------------------------------------------------------------------- #
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(CUR_DIR))
sys.path.append(PROJ_ROOT)                    # utils utils

from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

RES_DIR = "localExperiments/oriExperiments/model_result/cnnModelResult"
CKPT_DIR = "localExperiments/oriExperiments/model_result/cnn_checkPoints"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ------------------------------------------------------------------- #
# 1) 环境与日志
# ------------------------------------------------------------------- #
SEED = 2025
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(
    filename=os.path.join(RES_DIR, "cnn_experiment.log"),
    name="ORI_CNN_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)
logger.info(f"Running on: {DEVICE}  |  Random seed: {SEED}")

# ------------------------------------------------------------------- #
# 2) 读取 ORI 数据集
# ------------------------------------------------------------------- #
DATA_PATH = "data/ori/oriDataset_seq.npz"
logger.info(f"Loading dataset from: {DATA_PATH}")
data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_test,  y_test  = data["X_test"],  data["y_test"]
logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

NUM_CLASSES = int(np.max(y_train) + 1)        # 4

# ------------------------------------------------------------------- #
# 3) Dataset & DataLoader
# ------------------------------------------------------------------- #
class ORISeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 10, 1008)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64
dl_train = DataLoader(ORISeqDataset(X_train, y_train),
                      batch_size=BATCH_SIZE, shuffle=True)
dl_test  = DataLoader(ORISeqDataset(X_test,  y_test),
                      batch_size=BATCH_SIZE, shuffle=False)
logger.info(f"Batch size: {BATCH_SIZE}  |  Train batches: {len(dl_train)}  "
            f"Test batches: {len(dl_test)}")

# ------------------------------------------------------------------- #
# 4) CNN 模型
# ------------------------------------------------------------------- #
class CNN2D(nn.Module):
    """
    3-block 2-D CNN : 1×10×1008 → (Conv+BN+ReLU+Pool)*2 → Conv →
    GAP → fc(128→NUM_CLASSES)
    """
    def __init__(self, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),          # (10,1008)→(10,252)
            nn.Dropout(0.3)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),          # (10,252)→(10,63)
            nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(128, n_classes)

    def forward(self, x):                  # x: (B, 10, 1008)
        x = x.unsqueeze(1)                 # → (B,1,10,1008)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)         # (B,128)
        return self.fc(x)

model = CNN2D().to(DEVICE)
logger.info(f"CNN parameters: {sum(p.numel() for p in model.parameters())}")

# ------------------------------------------------------------------- #
# 5) 训练函数
# ------------------------------------------------------------------- #
def train(model, epochs=60):
    logger.info(f"Start training  |  Epochs: {epochs}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    hist = {"tr": [], "val": []}
    best_val = float("inf")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train(); tr_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(dl_train.dataset)

        # ---- val ----
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(dl_test.dataset)

        hist["tr"].append(tr_loss); hist["val"].append(val_loss)

        # checkpoint
        ckpt = os.path.join(CKPT_DIR, f"cnn_epoch_{ep:03d}.pth")
        is_best = val_loss < best_val
        save_checkpoint(model, optimizer, ep, val_loss,
                        filename=ckpt,
                        is_best=is_best,
                        best_filename=os.path.join(CKPT_DIR, "cnn_best.pth"))
        if is_best: best_val = val_loss

        if ep % 10 == 0 or ep <= 5:
            logger.info(f"Epoch {ep:03d} | TrainLoss {tr_loss:.4f}  "
                        f"ValLoss {val_loss:.4f}")

    logger.info(f"Training finished in {time.time() - t0:.1f} s  "
                f"| Best ValLoss {best_val:.4f}")

    torch.save(model.state_dict(), os.path.join(RES_DIR, "cnn_model.pth"))
    return hist

# ------------------------------------------------------------------- #
# 6) 评估函数
# ------------------------------------------------------------------- #
def evaluate(model):
    logger.info("Evaluating on test set")
    model.eval(); p_all, y_all = [], []

    # inference timing
    infer_t, samples = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(DEVICE)
            t1 = time.time()
            out = model(xb)
            infer_t += time.time() - t1
            p_all.append(out.cpu().argmax(1)); y_all.append(yb)
            samples += xb.size(0)

    y_pred = torch.cat(p_all).numpy()
    y_true = torch.cat(y_all).numpy()

    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rc  = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1  = f1_score(       y_true, y_pred, average="macro", zero_division=0)
    ips = samples / infer_t

    logger.info(f"Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  "
                f"F1:{f1:.4f}  Throughput:{ips:.2f} samples/s")

    # 保存分类报告
    rpt = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(RES_DIR, "cnn_classification_report.txt"), "w") as fp:
        fp.write("CNN Classification Report\n" + rpt)

    return acc, pr, rc, f1, y_true, y_pred

# ------------------------------------------------------------------- #
# 7) 主程序
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CNN training phase")
    logger.info("=" * 60)

    history = train(model, epochs=60)

    logger.info("=" * 60)
    logger.info("CNN evaluation phase")
    logger.info("=" * 60)

    acc, pr, rc, f1, y_true, y_pred = evaluate(model)
    with open(os.path.join(RES_DIR, "final_metrics.txt"), "w") as fp:
        fp.write("CNN Final Metrics\n")
        fp.write(f"Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  "
                 f"F1:{f1:.4f}\n")

    # 绘制 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history["tr"], label="Train Loss")
    plt.plot(history["val"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("CNN Training / Validation Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "cnn_loss_curves.png"), dpi=300); plt.close()

    # 混淆矩阵
    class_names = [f"Class-{i}" for i in range(NUM_CLASSES)]
    plt.figure(figsize=(8, 6))
    _ = plot_confusion_matrix(y_true, y_pred, class_names,
                              normalize=True, fontsize=14,
                              vmin=0, vmax=1, axis=1)
    plt.title("CNN Confusion Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "cnn_confusion_matrix.png"), dpi=300)
    plt.close()

    # 保存训练历史
    np.savez(os.path.join(RES_DIR, "training_history.npz"),
             cnn_train=history["tr"], cnn_val=history["val"])

    logger.info("=" * 60)
    logger.info("Experiment completed. All outputs in " + RES_DIR)
