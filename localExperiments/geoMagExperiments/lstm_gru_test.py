# Reproduce Table-5 LSTM / GRU baseline models 
import os, random, time, sys

# Fix the path to correctly import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import logger and checkpoint utilities
from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

# ------------------------ 0. Setup logging and result directory ------------- #
RESULT_DIR = "localExperiments/geoMagExperiments/model_result/seqModelResult"
CHECKPOINT_DIR = "localExperiments/geoMagExperiments/model_result/checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize logger
logger = get_logger(
    filename=os.path.join(RESULT_DIR, "lstm_gru_experiment.log"),
    name="LSTM_GRU_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)

# ------------------------ 1. Reproducibility ------------------------ #
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on: {DEVICE}")
logger.info(f"Random seed set to: {SEED}")
logger.info(f"Results will be saved to: {RESULT_DIR}")

# ------------------------ 2. Load dataset --------------------------- #
DATA_PATH = "data/geoMag/geoMagDataset.npz"
logger.info(f"Loading dataset from: {DATA_PATH}")

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_test"], data["y_test"]  

logger.info(f"Original data shapes - Train: {X_train.shape}, Val: {X_val.shape}")

# reshape flat 621-dim feature to (207,3) sequence --------------------------------
assert X_train.shape[1] == 621, "Expected 621 flat features"
X_train = X_train.reshape(-1, 207, 3)
X_val = X_val.reshape(-1, 207, 3)

logger.info(f"Reshaped to 3-axis time series - Train: {X_train.shape}, Val: {X_val.shape}")

# ------------------------ 3. Torch dataset -------------------------------------- #
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 16
dl_train = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
dl_val   = DataLoader(SeqDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

logger.info(f"DataLoaders created with batch size: {batch_size}")
logger.info(f"Train batches: {len(dl_train)}, Validation batches: {len(dl_val)}")

# ------------------------ 4. Models (Table-5) ----------------------------------- #
class RNNStack(nn.Module):
    """
    Three-layer LSTM / GRU as in Table-5.
    - Keep full sequence for first two layers
    - Apply BatchNorm1d on feature dimension (B, C, T)
    - Use last time-step only after the 3rd layer
    """
    def __init__(self, cell: str = "lstm"):
        super().__init__()
        RNN = nn.LSTM if cell.lower() == "lstm" else nn.GRU

        self.rnn1 = RNN(input_size=3,  hidden_size=32, batch_first=True)
        self.bn1  = nn.BatchNorm1d(32)
        self.dp1  = nn.Dropout(0.2)

        self.rnn2 = RNN(input_size=32, hidden_size=16, batch_first=True)
        self.bn2  = nn.BatchNorm1d(16)
        self.dp2  = nn.Dropout(0.2)

        self.rnn3 = RNN(input_size=16, hidden_size=8,  batch_first=True)
        self.bn3  = nn.BatchNorm1d(8)
        self.dp3  = nn.Dropout(0.2)

        self.fc   = nn.Linear(8, 3)

    def _bn_time(self, x, bn_layer):
        # x : (B,T,C)   -> (B,C,T) for BN -> back
        return bn_layer(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        o, _ = self.rnn1(x)                     # (B,T,32)
        o    = self.dp1(self._bn_time(o, self.bn1))

        o, _ = self.rnn2(o)                    # (B,T,16)
        o    = self.dp2(self._bn_time(o, self.bn2))

        o, _ = self.rnn3(o)                    # (B,T,8)
        o    = self.dp3(self._bn_time(o, self.bn3))

        out  = self.fc(o[:, -1, :])            # (B,3)
        return out

# instantiate
model_lstm = RNNStack(cell="lstm").to(DEVICE)
model_gru  = RNNStack(cell="gru" ).to(DEVICE)

logger.info("Models initialized:")
logger.info(f"LSTM parameters: {sum(p.numel() for p in model_lstm.parameters())}")
logger.info(f"GRU parameters: {sum(p.numel() for p in model_gru.parameters())}")

# ------------------------ 5. Training utils ------------------------------------- #
def train(model, name, num_epochs=200):
    logger.info(f"Starting training for {name} model")
    logger.info(f"Training parameters - Epochs: {num_epochs}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {"tr":[], "val":[]}
    epoch_times = []  # 添加时间记录
    best_val_loss = float("inf")  # 添加最佳验证损失跟踪

    start_time = time.time()
    
    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_train = 0.0
        for Xb, yb in dl_train:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * Xb.size(0)
        epoch_tr = running_train / len(dl_train.dataset)

        # Validation phase
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for Xb, yb in dl_val:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                out = model(Xb)
                loss = criterion(out, yb)
                running_val += loss.item() * Xb.size(0)
        epoch_val = running_val / len(dl_val.dataset)
        
        history["tr"].append(epoch_tr)
        history["val"].append(epoch_val)

        # Save checkpoint every epoch with best tracking
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name.lower()}_epoch_{epoch:03d}.pth")
        is_best = epoch_val < best_val_loss
        if is_best: 
            best_val_loss = epoch_val
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_val,
            filename=checkpoint_path,
            is_best=is_best,
            best_filename=os.path.join(CHECKPOINT_DIR, f"{name.lower()}_best.pth")
        )

        if epoch % 10 == 0 or epoch <= 5:
            logger.info(f"{name} | Epoch {epoch:03d}  TrainLoss {epoch_tr:.4f}  ValLoss {epoch_val:.4f}")
        
        epoch_times.append(time.time() - epoch_start)
    
    total_training_time = time.time() - start_time
    avg_epoch_time = float(np.mean(epoch_times))
    logger.info(f"{name} | Training completed in {total_training_time:.2f} seconds")
    logger.info(f"{name} | Average epoch time: {avg_epoch_time:.2f} seconds")
    
    # Save final model
    model_path = os.path.join(RESULT_DIR, f"{name.lower()}_model_final.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"{name} | Final model saved to: {model_path}")
    
    return history, total_training_time, avg_epoch_time

def evaluate(model, name):
    logger.info(f"Evaluating {name} model on validation set")
    
    model.eval()
    all_pred, all_true = [], []
    
    # 预热推理（避免第一次推理的初始化开销）
    with torch.no_grad():
        dummy_input = torch.randn(1, 207, 3).to(DEVICE)
        for _ in range(5):
            _ = model(dummy_input)
    
    # 计算推理时间
    inference_times = []
    total_samples = 0
    
    with torch.no_grad():
        for Xb, yb in dl_val:
            Xb = Xb.to(DEVICE)
            
            # 记录推理时间
            start_time = time.time()
            out = model(Xb)
            end_time = time.time()
            
            batch_time = end_time - start_time
            batch_size = Xb.size(0)
            inference_times.append(batch_time)
            total_samples += batch_size
            
            all_pred.append(out.cpu().argmax(1))
            all_true.append(yb)
    
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()
    
    # 计算平均推理时间
    total_inference_time = sum(inference_times)
    avg_inference_time_per_sample = total_inference_time / total_samples
    avg_inference_time_per_batch = total_inference_time / len(dl_val)
    
    # 记录推理时间信息
    logger.info(f"{name} Inference Timing:")
    logger.info(f"  Total inference time: {total_inference_time:.4f} seconds")
    logger.info(f"  Average time per sample: {avg_inference_time_per_sample*1000:.4f} ms")
    logger.info(f"  Average time per batch: {avg_inference_time_per_batch*1000:.4f} ms")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Throughput: {total_samples/total_inference_time:.2f} samples/second")
    
    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc  = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    f1  = f1_score(       y_true, y_pred, average="weighted", zero_division=0)
    
    # Log classification report
    class_report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"{name} Classification Report:\n{class_report}")
    
    # Save classification report to file
    report_path = os.path.join(RESULT_DIR, f"{name.lower()}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{name} Classification Report:\n")
        f.write(class_report)
    
    return acc, pr, rc, f1, y_true, y_pred, avg_inference_time_per_sample

# ------------------------ 6. Run training --------------------------------------- #
logger.info("=" * 50)
logger.info("Starting model training phase")
logger.info("=" * 50)

hist_lstm, total_time_lstm, avg_epoch_time_lstm = train(model_lstm, "LSTM")
hist_gru, total_time_gru, avg_epoch_time_gru = train(model_gru, "GRU")

# Load best checkpoints before evaluation (像CNN一样)
best_lstm_ckpt = os.path.join(CHECKPOINT_DIR, "lstm_best.pth")
best_gru_ckpt = os.path.join(CHECKPOINT_DIR, "gru_best.pth")

model_lstm.load_state_dict(torch.load(best_lstm_ckpt, map_location=DEVICE))
model_gru.load_state_dict(torch.load(best_gru_ckpt, map_location=DEVICE))
logger.info(f"Loaded best LSTM checkpoint from {best_lstm_ckpt}")
logger.info(f"Loaded best GRU checkpoint from {best_gru_ckpt}")

# ------------------------ 7. Evaluation ----------------------------------------- #
logger.info("=" * 50)
logger.info("Starting model evaluation phase")
logger.info("=" * 50)

metrics = {}
model_sizes = {}

for name, mdl, ckpt_path in [("LSTM", model_lstm, best_lstm_ckpt), ("GRU", model_gru, best_gru_ckpt)]:
    acc, pr, rc, f1, yt, yp, avg_inference_time = evaluate(mdl, name)
    
    # 计算模型大小 (像CNN一样)
    model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    model_sizes[name] = model_size_mb
    
    metrics[name] = (acc, pr, rc, f1, yt, yp, avg_inference_time)
    
    # 更详细的结果记录
    latency_ms = avg_inference_time * 1000
    throughput = 1 / avg_inference_time
    
    logger.info(f"{name} Final Results:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision: {pr:.4f}")
    logger.info(f"  Recall: {rc:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  Latency: {latency_ms:.2f} ms/sample")
    logger.info(f"  Throughput: {throughput:.2f} samples/second")
    logger.info(f"  Model Size: {model_size_mb:.2f} MB")

# Save comprehensive metrics to file (扩展版本)
metrics_path = os.path.join(RESULT_DIR, "final_metrics.txt")
with open(metrics_path, 'w') as f:
    f.write("Final Model Metrics\n")
    f.write("=" * 50 + "\n")
    
    # 添加训练时间信息
    f.write("TRAINING SUMMARY:\n")
    f.write(f"LSTM total train time (s): {total_time_lstm:.2f}\n")
    f.write(f"LSTM avg epoch time (s):   {avg_epoch_time_lstm:.2f}\n")
    f.write(f"GRU total train time (s):  {total_time_gru:.2f}\n")
    f.write(f"GRU avg epoch time (s):    {avg_epoch_time_gru:.2f}\n")
    f.write("-" * 30 + "\n")
    
    # 详细性能指标
    f.write("PERFORMANCE METRICS:\n")
    for name, (acc, pr, rc, f1, _, _, avg_inference_time) in metrics.items():
        f.write(f"{name} Results:\n")
        f.write(f"  Accuracy:                  {acc:.4f}\n")
        f.write(f"  Precision (weighted):      {pr:.4f}\n")
        f.write(f"  Recall (macro):            {rc:.4f}\n")
        f.write(f"  F1-score (weighted):       {f1:.4f}\n")
        f.write(f"  Latency per sample (ms):   {avg_inference_time*1000:.2f}\n")
        f.write(f"  Throughput (samples/s):    {1/avg_inference_time:.2f}\n")
        f.write(f"  Model size (MB):           {model_sizes[name]:.2f}\n")
        f.write(f"  Parameter count:           {sum(p.numel() for p in (model_lstm if name=='LSTM' else model_gru).parameters())}\n")
        f.write("-" * 30 + "\n")

logger.info(f"Final metrics saved to: {metrics_path}")

# ------------------------ 8. Plot loss curves ----------------------------------- #
logger.info("Generating loss curves plot")

# Plot LSTM curves
plt.figure(figsize=(10, 4))
plt.plot(hist_lstm["tr"], label="Train Loss", color='blue')
plt.plot(hist_lstm["val"], label="Validation Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training / Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
lstm_curve_path = os.path.join(RESULT_DIR, "lstm_curves.png")
plt.savefig(lstm_curve_path, dpi=300)
logger.info(f"LSTM curves saved to: {lstm_curve_path}")

# Plot GRU curves
plt.figure(figsize=(10, 4))
plt.plot(hist_gru["tr"], label="Train Loss", color='blue')
plt.plot(hist_gru["val"], label="Validation Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training / Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
gru_curve_path = os.path.join(RESULT_DIR, "gru_curves.png")
plt.savefig(gru_curve_path, dpi=300)
logger.info(f"GRU curves saved to: {gru_curve_path}")

# Plot combined comparison
plt.figure(figsize=(10, 6))
plt.plot(hist_lstm["tr"], label="LSTM-train", color='blue')
plt.plot(hist_lstm["val"], label="LSTM-val", color='blue', linestyle='--')
plt.plot(hist_gru["tr"], label="GRU-train", color='red')
plt.plot(hist_gru["val"], label="GRU-val", color='red', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
comparison_path = os.path.join(RESULT_DIR, "model_comparison.png")
plt.savefig(comparison_path, dpi=300)
logger.info(f"Model comparison saved to: {comparison_path}")

# ------------------------ 9. Confusion matrices --------------------------------- #
logger.info("Generating confusion matrices")

class_names = ["Light", "Medium", "Heavy"]

for name, (_,_,_,_, yt, yp, _) in metrics.items():
    plt.figure(figsize=(8, 6))
    ax = plot_confusion_matrix(
        test_y=yt, 
        pred_y=yp, 
        class_names=class_names, 
        normalize=True, 
        fontsize=16, 
        vmin=0, 
        vmax=1, 
        axis=1
    )
    plt.title(f"{name} Confusion Matrix", fontsize=18, pad=20)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULT_DIR, f"cm_{name.lower()}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    logger.info(f"{name} confusion matrix saved to: {cm_path}")

# Save training history
logger.info("Saving training history")
history_path = os.path.join(RESULT_DIR, "training_history.npz")
np.savez(history_path, 
         lstm_train=hist_lstm["tr"], lstm_val=hist_lstm["val"],
         gru_train=hist_gru["tr"], gru_val=hist_gru["val"])
logger.info(f"Training history saved to: {history_path}")

logger.info("=" * 50)
logger.info("Experiment completed successfully!")
logger.info(f"All results saved in: {RESULT_DIR}")
logger.info(f"Checkpoints saved in: {CHECKPOINT_DIR}")
logger.info("Generated files:")
logger.info("- lstm_gru_experiment.log (detailed log file)")
logger.info("- lstm_model_final.pth, gru_model_final.pth (final model weights)")
logger.info("- lstm_best.pth, gru_best.pth (best model checkpoints)")
logger.info("- final_metrics.txt (comprehensive summary metrics)")
logger.info("- lstm_curves.png, gru_curves.png (individual model curves)")
logger.info("- model_comparison.png (comparison plot)")
logger.info("- cm_lstm.png, cm_gru.png (confusion matrices)")
logger.info("- training_history.npz (training history data)")
logger.info("- lstm_classification_report.txt, gru_classification_report.txt")
logger.info("- Epoch checkpoints in checkPoints/ directory")
logger.info("=" * 50)
