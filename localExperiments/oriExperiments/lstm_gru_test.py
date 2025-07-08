# ori_lstm_gru_experiment.py
# -*- coding: utf-8 -*-
"""
Reproduce Table-5 style LSTM / GRU baseline on ORI sequence dataset
Dataset shape : Train (15019, 10, 1008) | Test (3560, 10, 1008)
Author: Efficient_Computing_System
"""

import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# 0) Setup project paths & utils import
# --------------------------------------------------------------------------- #
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)                      # for utils package

from utils.utils import (get_logger, save_checkpoint,
                         plot_confusion_matrix)    # same as upstream script

# --------------------------------------------------------------------------- #
# 1) Result directories & logger
# --------------------------------------------------------------------------- #
RESULT_DIR     = "localExperiments/oriExperiments/model_result/seqModelResult"
CHECKPOINT_DIR = "localExperiments/oriExperiments/model_result/checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger = get_logger(
    filename=os.path.join(RESULT_DIR, "ori_lstm_gru_experiment.log"),
    name="ORI_LSTM_GRU_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)

# --------------------------------------------------------------------------- #
# 2) Reproducibility
# --------------------------------------------------------------------------- #
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on: {DEVICE}")
logger.info(f"Random seed set to: {SEED}")

# --------------------------------------------------------------------------- #
# 3) Load ORI dataset
# --------------------------------------------------------------------------- #
DATA_PATH = "data/ori/oriDataset_seq.npz"
logger.info(f"Loading ORI dataset from: {DATA_PATH}")
data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_test,  y_test  = data["X_test"],  data["y_test"]
logger.info(f"Dataset loaded | Train: {X_train.shape}  Test: {X_test.shape}")

# Check label distribution
logger.info(f"y_train unique values: {np.unique(y_train)}")
logger.info(f"y_test unique values: {np.unique(y_test)}")
n_classes = len(np.unique(np.concatenate([y_train, y_test])))
logger.info(f"Number of classes: {n_classes}")

# --------------------------------------------------------------------------- #
# 4) Torch Dataset & DataLoader
# --------------------------------------------------------------------------- #
class ORISeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, T, 1008)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 64
dl_train = DataLoader(ORISeqDataset(X_train, y_train),
                      batch_size=BATCH_SIZE, shuffle=True)
dl_test  = DataLoader(ORISeqDataset(X_test,  y_test),
                      batch_size=BATCH_SIZE, shuffle=False)
logger.info(f"DataLoaders built | Train batches: {len(dl_train)}  "
            f"Test batches: {len(dl_test)}  Batch size: {BATCH_SIZE}")

# --------------------------------------------------------------------------- #
# 5) Model definitions
# --------------------------------------------------------------------------- #
class StackedRNN(nn.Module):
    """
    Three-layer LSTM / GRU for ORI dataset.
    Hidden sizes : 128 → 64 → 32
    BatchNorm1d is applied on feature dim after each layer.
    Final classifier uses last-step features.
    """
    def __init__(self, cell: str = "lstm", n_classes: int = 4):  # Changed from 2 to 4
        super().__init__()
        RNN = nn.LSTM if cell.lower() == "lstm" else nn.GRU

        self.rnn1 = RNN(input_size=1008, hidden_size=128,
                        batch_first=True)
        self.bn1  = nn.BatchNorm1d(128)
        self.dp1  = nn.Dropout(0.3)

        self.rnn2 = RNN(input_size=128, hidden_size=64,
                        batch_first=True)
        self.bn2  = nn.BatchNorm1d(64)
        self.dp2  = nn.Dropout(0.3)

        self.rnn3 = RNN(input_size=64, hidden_size=32,
                        batch_first=True)
        self.bn3  = nn.BatchNorm1d(32)
        self.dp3  = nn.Dropout(0.3)

        self.fc   = nn.Linear(32, n_classes)

    @staticmethod
    def _bn_time(x, bn_layer):
        # x: (B, T, C) -> (B, C, T) -> BN -> back
        return bn_layer(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        out, _ = self.rnn1(x)              # (B, T, 128)
        out    = self.dp1(self._bn_time(out, self.bn1))

        out, _ = self.rnn2(out)            # (B, T, 64)
        out    = self.dp2(self._bn_time(out, self.bn2))

        out, _ = self.rnn3(out)            # (B, T, 32)
        out    = self.dp3(self._bn_time(out, self.bn3))

        logits = self.fc(out[:, -1, :])    # (B, n_classes)
        return logits

model_lstm = StackedRNN(cell="lstm", n_classes=n_classes).to(DEVICE)  # Use detected n_classes
model_gru  = StackedRNN(cell="gru", n_classes=n_classes).to(DEVICE)   # Use detected n_classes

logger.info("Models initialized")
logger.info(f"LSTM parameters: {sum(p.numel() for p in model_lstm.parameters())}")
logger.info(f"GRU  parameters: {sum(p.numel() for p in model_gru.parameters())}")

# --------------------------------------------------------------------------- #
# 6) Training & evaluation helpers
# --------------------------------------------------------------------------- #
def train(model, name, epochs=60):
    logger.info(f"Starting training for {name}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=1e-4)
    history = {"tr": [], "val": []}
    start = time.time()

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train(); running_tr = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            running_tr += loss.item() * xb.size(0)
        ep_tr = running_tr / len(dl_train.dataset)

        # ---- val ----
        model.eval(); running_val = 0.0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                running_val += criterion(model(xb), yb).item() * xb.size(0)
        ep_val = running_val / len(dl_test.dataset)

        history["tr"].append(ep_tr)
        history["val"].append(ep_val)

        # save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR,
                                 f"{name.lower()}_epoch_{ep:03d}.pth")
        save_checkpoint(model=model, optimizer=optimizer,
                        epoch=ep, loss=ep_val, filename=ckpt_path)

        if ep % 10 == 0 or ep <= 5:
            logger.info(f"{name} | Epoch {ep:03d} "
                        f"TrainLoss {ep_tr:.4f}  ValLoss {ep_val:.4f}")

    elapsed = time.time() - start
    logger.info(f"{name} | Training finished in {elapsed:.1f} s")

    # save final weights
    final_path = os.path.join(RESULT_DIR, f"{name.lower()}_model.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"{name} | Final model saved to: {final_path}")
    return history

def evaluate(model, name):
    logger.info(f"Evaluating {name}")
    model.eval(); all_pred, all_true = [], []

    # warm-up (avoid first-call overhead on GPU)
    with torch.no_grad():
        _ = model(torch.randn(1, 10, 1008).to(DEVICE))

    # inference timing
    t0 = time.time()
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(DEVICE)
            out = model(xb)
            all_pred.append(out.cpu().argmax(1))
            all_true.append(yb)
    total_t = time.time() - t0
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    acc = accuracy_score(y_true, y_pred)
    # Changed from 'binary' to 'macro' for multiclass
    pr  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rc  = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1  = f1_score(       y_true, y_pred, average="macro", zero_division=0)
    ips = len(y_true) / total_t

    logger.info(f"{name} | Acc:{acc:.4f}  Prec:{pr:.4f} "
                f"Recall:{rc:.4f}  F1:{f1:.4f}  "
                f"Throughput:{ips:.2f} samples/s")

    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"{name} Classification Report:\n{report}")
    with open(os.path.join(RESULT_DIR,
              f"{name.lower()}_classification_report.txt"), "w") as fp:
        fp.write(f"{name} Classification Report\n{report}")

    return acc, pr, rc, f1, y_true, y_pred, ips

# --------------------------------------------------------------------------- #
# 7) Run experiment
# --------------------------------------------------------------------------- #
logger.info("=" * 60)
logger.info("Starting training phase")
logger.info("=" * 60)
hist_lstm = train(model_lstm, "LSTM")
hist_gru  = train(model_gru,  "GRU")

logger.info("=" * 60)
logger.info("Starting evaluation phase")
logger.info("=" * 60)
metrics = {}
for nm, mdl in [("LSTM", model_lstm), ("GRU", model_gru)]:
    metrics[nm] = evaluate(mdl, nm)

# --------------------------------------------------------------------------- #
# 8) Save global metrics
# --------------------------------------------------------------------------- #
with open(os.path.join(RESULT_DIR, "final_metrics.txt"), "w") as fp:
    fp.write("Final Model Metrics\n" + "=" * 50 + "\n")
    for nm, (acc, pr, rc, f1, _, _, ips) in metrics.items():
        fp.write(f"{nm} | Acc:{acc:.4f}  Prec:{pr:.4f}  "
                 f"Recall:{rc:.4f}  F1:{f1:.4f}\n")
        fp.write(f"{nm} | Throughput:{ips:.2f} samples/s\n")
        fp.write("-" * 50 + "\n")
logger.info("Metric file saved")

# --------------------------------------------------------------------------- #
# 9) Draw loss curves
# --------------------------------------------------------------------------- #
def _plot_curve(hist, title, path):
    plt.figure(figsize=(10, 4))
    plt.plot(hist["tr"], label="Train Loss")
    plt.plot(hist["val"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

_plot_curve(hist_lstm, "LSTM Training / Validation Loss",
            os.path.join(RESULT_DIR, "lstm_curves.png"))
_plot_curve(hist_gru,  "GRU Training / Validation Loss",
            os.path.join(RESULT_DIR, "gru_curves.png"))

plt.figure(figsize=(10, 6))
plt.plot(hist_lstm["tr"], label="LSTM-train")
plt.plot(hist_lstm["val"], label="LSTM-val", linestyle="--")
plt.plot(hist_gru["tr"],  label="GRU-train")
plt.plot(hist_gru["val"], label="GRU-val",  linestyle="--")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training / Validation Loss Comparison")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "model_comparison.png"), dpi=300)
plt.close()
logger.info("Loss curves plotted")

# --------------------------------------------------------------------------- #
# 10) Confusion matrices
# --------------------------------------------------------------------------- #
logger.info("Drawing confusion matrices")
# Updated class names for 4-class classification
class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]  # Adjust based on your actual class meanings
for nm, (_,_,_,_, yt, yp, _) in metrics.items():
    ax = plot_confusion_matrix(test_y=yt, pred_y=yp,
                               class_names=class_names,
                               normalize=True, fontsize=14,
                               vmin=0, vmax=1, axis=1)
    plt.title(f"{nm} Confusion Matrix", pad=20)
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, f"cm_{nm.lower()}.png")
    plt.savefig(cm_path, dpi=300); plt.close()
    logger.info(f"{nm} confusion matrix saved to: {cm_path}")

# --------------------------------------------------------------------------- #
# 11) Save training history
# --------------------------------------------------------------------------- #
np.savez(os.path.join(RESULT_DIR, "training_history.npz"),
         lstm_train=hist_lstm["tr"], lstm_val=hist_lstm["val"],
         gru_train=hist_gru["tr"],  gru_val=hist_gru["val"])
logger.info("Training history saved")

logger.info("=" * 60)
logger.info("Experiment completed successfully!")
logger.info(f"All results stored in: {RESULT_DIR}")
logger.info(f"Checkpoints stored in: {CHECKPOINT_DIR}")
logger.info("=" * 60)
