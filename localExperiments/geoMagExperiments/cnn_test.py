import os, random, time, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------- #
# 0. Path & logger
# ------------------------------------------------------------------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

RESULT_DIR     = "localExperiments/geoMagExperiments/model_result/cnnModelResult"
CHECKPOINT_DIR = "localExperiments/geoMagExperiments/model_result/cnn_checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger = get_logger(
    filename=os.path.join(RESULT_DIR, "cnn_experiment.log"),
    name="CNN_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)

# ------------------------------------------------------------------------- #
# 1. Reproducibility
# ------------------------------------------------------------------------- #
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {DEVICE}")
logger.info(f"Random seed: {SEED}")

# ------------------------------------------------------------------------- #
# 2. Load data
# ------------------------------------------------------------------------- #
DATA_PATH = "data/geoMag/geoMagDataset.npz"
logger.info(f"Loading dataset from {DATA_PATH}")

data      = np.load(DATA_PATH)
X_train   = data["X_train"]
y_train   = data["y_train"]
X_val     = data["X_test"]
y_val     = data["y_test"]

assert X_train.shape[1] == 621, "Expected 621 flat features"
X_train = X_train.reshape(-1, 207, 3)
X_val   = X_val.reshape(-1, 207, 3)
logger.info(f"Train shape {X_train.shape} | Val shape {X_val.shape}")

# ------------------------------------------------------------------------- #
# 3. Dataset
# ------------------------------------------------------------------------- #
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

BATCH_SIZE = 16
dl_train = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
dl_val   = DataLoader(SeqDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
logger.info(f"Batch size {BATCH_SIZE} | Train batches {len(dl_train)} | Val batches {len(dl_val)}")

# ------------------------------------------------------------------------- #
# 4. Model
# ------------------------------------------------------------------------- #
class CNN(nn.Module):
    def __init__(self, input_size=3, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.dp1   = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.dp2   = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(32)
        self.dp3   = nn.Dropout(0.2)

        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(32, num_classes)

    def forward(self, x):          # x: (B, T, C)
        x = x.transpose(1, 2)      # -> (B, C, T)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dp2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dp3(x)

        x = self.gap(x).squeeze(-1)  # (B, C)
        return self.fc(x)

model = CNN().to(DEVICE)
PARAMS_NUM = sum(p.numel() for p in model.parameters())
logger.info(f"Model parameters: {PARAMS_NUM}")

# ------------------------------------------------------------------------- #
# 5. Training
# ------------------------------------------------------------------------- #
def train(model, num_epochs: int = 200):
    logger.info(f"Start training for {num_epochs} epochs")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history, epoch_times = {"tr": [], "val": []}, []

    overall_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # -------- train phase -------- #
        model.train()
        tr_loss_sum = 0.0
        for Xb, yb in dl_train:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * Xb.size(0)
        epoch_tr = tr_loss_sum / len(dl_train.dataset)

        # -------- validation phase -------- #
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for Xb, yb in dl_val:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                loss   = criterion(logits, yb)
                val_loss_sum += loss.item() * Xb.size(0)
        epoch_val = val_loss_sum / len(dl_val.dataset)

        history["tr"].append(epoch_tr)
        history["val"].append(epoch_val)

        # -------- checkpoint -------- #
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"cnn_epoch_{epoch:03d}.pth")
        is_best   = epoch_val < best_val_loss
        if is_best: best_val_loss = epoch_val
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_val,
            filename=ckpt_path,
            is_best=is_best,
            best_filename=os.path.join(CHECKPOINT_DIR, "cnn_best.pth")
        )

        if epoch % 10 == 0 or epoch <= 5:
            logger.info(f"Epoch {epoch:03d} | TrainLoss {epoch_tr:.4f} | ValLoss {epoch_val:.4f}")

        epoch_times.append(time.time() - epoch_start)

    total_train_time   = time.time() - overall_start
    avg_epoch_time     = float(np.mean(epoch_times))
    logger.info(f"Total train time: {total_train_time:.2f}s | Avg epoch time: {avg_epoch_time:.2f}s")

    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "cnn_model_final.pth"))
    return history, total_train_time, avg_epoch_time

# ------------------------------------------------------------------------- #
# 6. Evaluation
# ------------------------------------------------------------------------- #
def evaluate(model):
    logger.info("Evaluating best checkpoint")

    # ----- warm-up inference to amortize one-time CUDA cost ----- #
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 207, 3).to(DEVICE)
        for _ in range(5): _ = model(dummy)

    # ----- timing ----- #
    total_inf_time = 0.0
    total_samples  = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for Xb, yb in dl_val:
            Xb = Xb.to(DEVICE)
            start = time.time()
            out   = model(Xb)
            end   = time.time()

            total_inf_time += (end - start)
            total_samples  += Xb.size(0)

            y_pred.append(out.cpu().argmax(1))
            y_true.append(yb)

    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()

    avg_time_per_sample = total_inf_time / total_samples
    avg_time_per_batch  = total_inf_time / len(dl_val)

    logger.info(f"Inference | total {total_inf_time:.4f}s | "
                f"per-sample {avg_time_per_sample*1000:.3f}ms | "
                f"per-batch {avg_time_per_batch*1000:.3f}ms | "
                f"throughput {total_samples/total_inf_time:.2f} samples/s")

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")

    class_report = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(RESULT_DIR, "cnn_classification_report.txt"), "w") as fp:
        fp.write(class_report)

    return acc, f1, avg_time_per_sample, total_inf_time, y_true, y_pred

# ------------------------------------------------------------------------- #
# 7. Main
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("="*60 + "\nTRAINING PHASE\n" + "="*60)
    hist, total_t, avg_epoch_t = train(model, num_epochs=200)

    # ---- load best checkpoint before evaluation ---- #
    best_ckpt = os.path.join(CHECKPOINT_DIR, "cnn_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    logger.info(f"Loaded best checkpoint from {best_ckpt}")

    logger.info("="*60 + "\nEVALUATION PHASE\n" + "="*60)
    acc, f1, latency_s, inf_time_s, y_true, y_pred = evaluate(model)

    latency_ms = latency_s * 1000.0
    model_size_mb = os.path.getsize(best_ckpt) / (1024 * 1024)

    logger.info(f"Final | ACC {acc:.4f} | F1 {f1:.4f} | "
                f"Latency {latency_ms:.2f}ms | "
                f"InfTime {inf_time_s:.2f}s | "
                f"ModelSize {model_size_mb:.2f}MB | Params {PARAMS_NUM}")

    # save metrics
    with open(os.path.join(RESULT_DIR, "final_metrics.txt"), "w") as fp:
        fp.write("CNN Final Metrics\n")
        fp.write("-"*50 + "\n")
        fp.write(f"Total train time (s):      {total_t:.2f}\n")
        fp.write(f"Average epoch time (s):    {avg_epoch_t:.2f}\n")
        fp.write(f"Accuracy:                  {acc:.4f}\n")
        fp.write(f"F1-score (weighted):       {f1:.4f}\n")
        fp.write(f"Latency per sample (ms):   {latency_ms:.2f}\n")
        fp.write(f"Total inference time (s):  {inf_time_s:.2f}\n")
        fp.write(f"Model size (MB):           {model_size_mb:.2f}\n")
        fp.write(f"Parameter count:           {PARAMS_NUM}\n")
    
    # ------------------------ 8. Plot loss curves ------------------------ #
    logger.info("Generating loss curves plot")

    # 设置seaborn样式（仿照temporaryPlotFile风格）
    import seaborn as sns
    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 8))
    # 使用与temporaryPlotFile相同的蓝色和橙色配色
    plt.plot(hist["tr"], label="Train Loss", color='#3498db', linewidth=3, marker='o', markersize=6, markerfacecolor='#2980b9', markeredgecolor='white', markeredgewidth=2)
    plt.plot(hist["val"], label="Validation Loss", color='#f39c12', linewidth=3, marker='s', markersize=6, markerfacecolor='#e67e22', markeredgecolor='white', markeredgewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14, color='#2c3e50')
    plt.title("CNN Training / Validation Loss", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    curve_path = os.path.join(RESULT_DIR, "cnn_loss_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"CNN loss curves saved to: {curve_path}")
    
    # ------------------------ 9. Confusion matrix ------------------------ #
    logger.info("Generating confusion matrix")
    
    class_names = ["Light", "Medium", "Heavy"]
    
    plt.figure(figsize=(8, 6))
    ax = plot_confusion_matrix(
        test_y=y_true, 
        pred_y=y_pred, 
        class_names=class_names, 
        normalize=True, 
        fontsize=16, 
        vmin=0, 
        vmax=1, 
        axis=1
    )
    plt.title("CNN Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULT_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"CNN confusion matrix saved to: {cm_path}")
    
    # Save training history
    logger.info("Saving training history")
    history_path = os.path.join(RESULT_DIR, "training_history.npz")
    np.savez(history_path, 
             cnn_train=hist["tr"], 
             cnn_val=hist["val"])
    logger.info(f"Training history saved to: {history_path}")
    
    logger.info("=" * 50)
    logger.info("CNN experiment completed successfully!")
    logger.info(f"All results saved in: {RESULT_DIR}")
    logger.info(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    logger.info("Generated files:")
    logger.info("- cnn_experiment.log (log file)")
    logger.info("- cnn_model.pth (final model weights)")
    logger.info("- cnn_best.pth (best model checkpoint)")
    logger.info("- final_metrics.txt (summary metrics)")
    logger.info("- cnn_loss_curves.png (loss curves)")
    logger.info("- cnn_confusion_matrix.png (confusion matrix)")
    logger.info("- training_history.npz (training history data)")
    logger.info("- cnn_classification_report.txt (classification report)")
    logger.info("- Epoch checkpoints in cnn_checkPoints/ directory")
    logger.info("=" * 50)