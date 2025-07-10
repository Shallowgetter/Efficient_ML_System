"""
CNN backbone with Spiking Neurons for GeoMagnetic Dataset
"""

import os, random, time, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from spikingjelly.activation_based import layer, neuron, functional, surrogate, encoding
from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

# ---------- 0. Setup directories & logger ----------
RESULT_DIR = "localExperiments/geoMagExperiments/model_result/snnModelResult"
CHECKPOINT_DIR = "localExperiments/geoMagExperiments/model_result/snn_checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger = get_logger(
    filename=os.path.join(RESULT_DIR, "snn_experiment.log"),
    name="SNN_Experiment",
    level="INFO",
    overwrite=True,
    to_stdout=True
)

# ---------- 1. Reproducibility ----------
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
logger.info(f"Results will be saved to: {RESULT_DIR}")
logger.info(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

# ---------- 2. Load and preprocess data ----------
DATA_PATH = "data/geoMag/geoMagDataset.npz"
logger.info(f"Loading dataset from: {DATA_PATH}")

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_test"], data["y_test"]

logger.info(f"Original data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
assert X_train.shape[1] == 621, "Expected 621 flat features"
X_train = X_train.reshape(-1, 207, 3)  # (B, T, C)
X_val = X_val.reshape(-1, 207, 3)
logger.info(f"Reshaped to 3-axis time series - Train: {X_train.shape}, Val: {X_val.shape}")

# ---------- 3. Dataset & DataLoader ----------
class SeqDataset(Dataset):
    """Tensor wrapper for sequence data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

BATCH_SIZE = 16
dl_train = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Train batches: {len(dl_train)}, Val batches: {len(dl_val)}")

# ---------- 4. Spiking Network Definition ----------
class SNN(nn.Module):
    """1-D CNN backbone with IF neurons."""
    def __init__(self, input_channels=3, num_classes=3, time_steps=16):
        super().__init__()
        self.T = time_steps
        
        self.conv1 = layer.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dp1 = nn.Dropout(0.2)

        self.conv2 = layer.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dp2 = nn.Dropout(0.2)

        self.conv3 = layer.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dp3 = nn.Dropout(0.2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        x shape: (B, T_len, C_in) -> transpose to (B, C_in, T_len)
        Simulate for self.T steps; reuse same analog input each step (rate coding).
        """
        x = x.transpose(1, 2)  # (B, C, T)
        out_spk_sum = 0
        
        for _ in range(self.T):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.sn1(out)
            out = self.dp1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.sn2(out)
            out = self.dp2(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.sn3(out)
            out = self.dp3(out)

            out = self.gap(out).squeeze(-1)   # (B, 32)
            out = self.fc(out)                # (B, num_classes)
            out_spk_sum += out

        return out_spk_sum / self.T          # average output

model = SNN(time_steps=16).to(DEVICE)
PARAMS_NUM = sum(p.numel() for p in model.parameters())
logger.info(f"SNN model initialized with {PARAMS_NUM} parameters")
logger.info(f"SNN time steps: {model.T}")

# ---------- 5. Training Function ----------
def train(net, num_epochs=200, lr=1e-3):
    logger.info(f"Starting SNN training for {num_epochs} epochs")
    logger.info(f"Learning rate: {lr}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {"tr": [], "val": []}
    epoch_times = []
    best_val_loss = float("inf")
    
    overall_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # -------- Training phase --------
        net.train()
        tr_loss_sum = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            functional.reset_net(net)  # Clear membrane potentials
            
            logits = net(xb)  # Forward pass with internal time steps
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * xb.size(0)
        
        epoch_tr = tr_loss_sum / len(dl_train.dataset)

        # -------- Validation phase --------
        net.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                functional.reset_net(net)
                
                logits = net(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
        
        epoch_val = val_loss_sum / len(dl_val.dataset)

        history["tr"].append(epoch_tr)
        history["val"].append(epoch_val)

        # -------- Checkpoint management --------
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"snn_epoch_{epoch:03d}.pth")
        is_best = epoch_val < best_val_loss
        if is_best: 
            best_val_loss = epoch_val
        
        save_checkpoint(
            model=net,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_val,
            filename=ckpt_path,
            is_best=is_best,
            best_filename=os.path.join(CHECKPOINT_DIR, "snn_best.pth")
        )

        if epoch % 10 == 0 or epoch <= 5:
            logger.info(f"SNN | Epoch {epoch:03d} | TrainLoss {epoch_tr:.4f} | ValLoss {epoch_val:.4f}")
        
        if is_best:
            logger.info(f"SNN | New best validation loss at epoch {epoch}: {epoch_val:.4f}")
        
        epoch_times.append(time.time() - epoch_start)

    total_train_time = time.time() - overall_start
    avg_epoch_time = float(np.mean(epoch_times))
    logger.info(f"SNN training completed in {total_train_time:.2f} seconds")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(RESULT_DIR, "snn_model_final.pth")
    torch.save(net.state_dict(), final_model_path)
    logger.info(f"Final SNN model saved to: {final_model_path}")
    
    return history, total_train_time, avg_epoch_time

# ---------- 6. Evaluation Function ----------
def evaluate(net):
    logger.info("Evaluating SNN model on validation set")
    
    net.eval()
    
    # Warm-up inference
    with torch.no_grad():
        dummy_input = torch.randn(1, 207, 3).to(DEVICE)
        for _ in range(5):
            functional.reset_net(net)
            _ = net(dummy_input)
    
    # Measure inference time and collect predictions
    inference_times = []
    total_samples = 0
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(DEVICE)
            
            # Measure inference time
            start_time = time.time()
            functional.reset_net(net)
            logits = net(xb)
            end_time = time.time()
            
            batch_time = end_time - start_time
            batch_size = xb.size(0)
            inference_times.append(batch_time)
            total_samples += batch_size
            
            y_pred.append(logits.cpu().argmax(1))
            y_true.append(yb)
    
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    
    # Calculate timing metrics
    total_inference_time = sum(inference_times)
    avg_inference_time_per_sample = total_inference_time / total_samples
    avg_inference_time_per_batch = total_inference_time / len(dl_val)
    
    logger.info(f"SNN Inference Timing:")
    logger.info(f"  Total inference time: {total_inference_time:.4f} seconds")
    logger.info(f"  Average time per sample: {avg_inference_time_per_sample*1000:.4f} ms")
    logger.info(f"  Average time per batch: {avg_inference_time_per_batch*1000:.4f} ms")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Throughput: {total_samples/total_inference_time:.2f} samples/second")
    
    # Calculate performance metrics
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    logger.info(f"SNN Final Results:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision (weighted): {pr:.4f}")
    logger.info(f"  Recall (macro): {rc:.4f}")
    logger.info(f"  F1-score (weighted): {f1:.4f}")
    logger.info(f"  Latency: {avg_inference_time_per_sample*1000:.2f} ms/sample")
    logger.info(f"  Throughput: {total_samples/total_inference_time:.2f} samples/second")
    
    # Classification report
    class_report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"SNN Classification Report:\n{class_report}")
    
    # Save classification report
    report_path = os.path.join(RESULT_DIR, "snn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("SNN Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(class_report)
    logger.info(f"Classification report saved to: {report_path}")
    
    return acc, pr, rc, f1, y_true, y_pred, avg_inference_time_per_sample

# ---------- 7. Main Execution ----------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TRAINING PHASE")
    logger.info("=" * 60)
    
    hist, total_train_time, avg_epoch_time = train(model, num_epochs=200)
    
    # Load best checkpoint before evaluation
    best_ckpt = os.path.join(CHECKPOINT_DIR, "snn_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    logger.info(f"Loaded best checkpoint from {best_ckpt}")
    
    logger.info("=" * 60)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 60)
    
    acc, pr, rc, f1, y_true, y_pred, avg_inference_time = evaluate(model)
    
    # Calculate additional metrics
    latency_ms = avg_inference_time * 1000
    model_size_mb = os.path.getsize(best_ckpt) / (1024 * 1024)
    
    logger.info(f"SNN Summary | ACC {acc:.4f} | F1 {f1:.4f} | "
                f"Latency {latency_ms:.2f}ms | "
                f"ModelSize {model_size_mb:.2f}MB | Params {PARAMS_NUM}")
    
    # ---------- 8. Save comprehensive metrics ----------
    logger.info("Saving comprehensive metrics summary")
    
    metrics_path = os.path.join(RESULT_DIR, "final_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("SNN Model Final Metrics\n")
        f.write("=" * 50 + "\n")
        
        f.write("TRAINING SUMMARY:\n")
        f.write(f"Total train time (s):          {total_train_time:.2f}\n")
        f.write(f"Average epoch time (s):        {avg_epoch_time:.2f}\n")
        f.write(f"Time steps:                    {model.T}\n")
        f.write("-" * 30 + "\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Accuracy:                      {acc:.4f}\n")
        f.write(f"Precision (weighted):          {pr:.4f}\n")
        f.write(f"Recall (macro):                {rc:.4f}\n")
        f.write(f"F1-score (weighted):           {f1:.4f}\n")
        f.write("-" * 30 + "\n")
        
        f.write("EFFICIENCY METRICS:\n")
        f.write(f"Latency per sample (ms):       {latency_ms:.4f}\n")
        f.write(f"Throughput (samples/s):        {1/avg_inference_time:.2f}\n")
        f.write(f"Model size (MB):               {model_size_mb:.2f}\n")
        f.write(f"Parameter count:               {PARAMS_NUM}\n")
        f.write("-" * 30 + "\n")
        
        f.write("SNN SPECIFIC:\n")
        f.write(f"Surrogate function:            ATan\n")
        f.write(f"Neuron type:                   IFNode\n")
        f.write(f"Encoding:                      Rate coding\n")
        f.write(f"Random seed:                   {SEED}\n")
    
    logger.info(f"Final metrics saved to: {metrics_path}")
    
    # ---------- 9. Plot loss curves ----------
    logger.info("Generating loss curves plot")
    
    import seaborn as sns
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 8))
    plt.plot(hist["tr"], label="Train Loss", color='#3498db', linewidth=3, marker='o', markersize=6, 
             markerfacecolor='#2980b9', markeredgecolor='white', markeredgewidth=2)
    plt.plot(hist["val"], label="Validation Loss", color='#e74c3c', linewidth=3, marker='s', markersize=6, 
             markerfacecolor='#c0392b', markeredgecolor='white', markeredgewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14, color='#2c3e50')
    plt.title("SNN Training / Validation Loss", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    curve_path = os.path.join(RESULT_DIR, "snn_loss_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"SNN loss curves saved to: {curve_path}")
    
    # ---------- 10. Confusion matrix ----------
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
    plt.title("SNN Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULT_DIR, "snn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"SNN confusion matrix saved to: {cm_path}")
    
    # ---------- 11. Save training history ----------
    logger.info("Saving training history")
    history_path = os.path.join(RESULT_DIR, "training_history.npz")
    np.savez(history_path, 
             snn_train=hist["tr"],
             snn_val=hist["val"])
    logger.info(f"Training history saved to: {history_path}")
    
    # ---------- 12. Final summary ----------
    logger.info("=" * 50)
    logger.info("SNN experiment completed successfully!")
    logger.info(f"All results saved in: {RESULT_DIR}")
    logger.info(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    logger.info("Generated files:")
    logger.info("- snn_experiment.log (detailed log file)")
    logger.info("- snn_model_final.pth (final model weights)")
    logger.info("- snn_best.pth (best model checkpoint)")
    logger.info("- final_metrics.txt (comprehensive summary metrics)")
    logger.info("- snn_loss_curves.png (loss curves)")
    logger.info("- snn_confusion_matrix.png (confusion matrix)")
    logger.info("- training_history.npz (training history data)")
    logger.info("- snn_classification_report.txt (classification report)")
    logger.info("- Epoch checkpoints in snn_checkPoints/ directory")
    logger.info("=" * 50)
    
    logger.info("Key Results Summary:")
    logger.info(f"- Accuracy: {acc:.4f}")
    logger.info(f"- F1-Score: {f1:.4f}")
    logger.info(f"- Training Time: {total_train_time:.2f}s")
    logger.info(f"- Inference Latency: {latency_ms:.2f}ms")
    logger.info(f"- Model Size: {model_size_mb:.2f}MB")
    logger.info(f"- Time Steps: {model.T}")
    logger.info("=" * 50)
