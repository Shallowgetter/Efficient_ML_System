import os, random, time, sys

# Fix the path to correctly import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import logger and checkpoint utilities
from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

# ------------------------ 0. Setup logging and result directory ------------- #
RESULT_DIR = "localExperiments/geoMagExperiments/model_result/cnnModelResult"
CHECKPOINT_DIR = "localExperiments/geoMagExperiments/model_result/cnn_checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize logger
logger = get_logger(
    filename=os.path.join(RESULT_DIR, "cnn_experiment.log"),
    name="CNN_Experiment",
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on: {DEVICE}")
logger.info(f"Random seed set to: {SEED}")
logger.info(f"Results will be saved to: {RESULT_DIR}")

# ------------------------ 2. Load and preprocess data ------------------------ #
DATA_PATH = "data/geoMag/geoMagDataset.npz"
logger.info(f"Loading dataset from: {DATA_PATH}")

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_test"], data["y_test"]  

logger.info(f"Original data shapes - Train: {X_train.shape}, Val: {X_val.shape}")

assert X_train.shape[1] == 621, "Expected 621 flat features"
X_train = X_train.reshape(-1, 207, 3)
X_val = X_val.reshape(-1, 207, 3)

logger.info(f"Reshaped to 3-axis time series - Train: {X_train.shape}, Val: {X_val.shape}")

# ------------------------ 3. Dataset and DataLoader ------------------------ #
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

# ------------------------ 4. CNN Model Definition ------------------------ #
class CNN(nn.Module):
    def __init__(self, input_size=3, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dp1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.dp3 = nn.Dropout(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dp3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        
        out = self.fc(x)
        return out

model = CNN().to(DEVICE)
logger.info("CNN model initialized")
logger.info(f"CNN parameters: {sum(p.numel() for p in model.parameters())}")

# ------------------------ 5. Training Function ------------------------ #
def train(model, num_epochs=200):
    logger.info("Starting CNN training")
    logger.info(f"Training parameters - Epochs: {num_epochs}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {"tr": [], "val": []}
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
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

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cnn_epoch_{epoch:03d}.pth")
        is_best = epoch_val < best_val_loss
        if is_best:
            best_val_loss = epoch_val
        
        best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "cnn_best.pth")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_val,
            filename=checkpoint_path,
            is_best=is_best,
            best_filename=best_checkpoint_path
        )

        if epoch % 10 == 0 or epoch <= 5:
            logger.info(f"CNN | Epoch {epoch:03d}  TrainLoss {epoch_tr:.4f}  ValLoss {epoch_val:.4f}")
        
        if is_best:
            logger.info(f"CNN | New best model at epoch {epoch} with val_loss: {epoch_val:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"CNN | Training completed in {training_time:.2f} seconds")
    logger.info(f"CNN | Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    model_path = os.path.join(RESULT_DIR, "cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"CNN | Model saved to: {model_path}")
    
    return history

# ------------------------ 6. Evaluation Function ------------------------ #
def evaluate(model):
    logger.info("Evaluating CNN model on validation set")
    
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
    logger.info(f"CNN Inference Timing:")
    logger.info(f"  Total inference time: {total_inference_time:.4f} seconds")
    logger.info(f"  Average time per sample: {avg_inference_time_per_sample*1000:.4f} ms")
    logger.info(f"  Average time per batch: {avg_inference_time_per_batch*1000:.4f} ms")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Throughput: {total_samples/total_inference_time:.2f} samples/second")
    
    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Log classification report
    class_report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"CNN Classification Report:\n{class_report}")
    
    # Save classification report to file
    report_path = os.path.join(RESULT_DIR, "cnn_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("CNN Classification Report:\n")
        f.write(class_report)
    
    return acc, pr, rc, f1, y_true, y_pred, avg_inference_time_per_sample

# ------------------------ 7. Main Execution ------------------------ #
if __name__ == '__main__':
    # Run training
    logger.info("=" * 50)
    logger.info("Starting CNN model training phase")
    logger.info("=" * 50)
    
    hist_cnn = train(model, num_epochs=200)
    
    # Run evaluation
    logger.info("=" * 50)
    logger.info("Starting CNN model evaluation phase")
    logger.info("=" * 50)
    
    acc, pr, rc, f1, y_true, y_pred = evaluate(model)
    logger.info(f"CNN Final Results | Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  F1:{f1:.4f}")
    
    # Save metrics to file
    metrics_path = os.path.join(RESULT_DIR, "final_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("CNN Model Final Metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"CNN | Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  F1:{f1:.4f}\n")
    
    logger.info(f"Final metrics saved to: {metrics_path}")
    
    # ------------------------ 8. Plot loss curves ------------------------ #
    logger.info("Generating loss curves plot")
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_cnn["tr"], label="Train Loss", color='blue')
    plt.plot(hist_cnn["val"], label="Validation Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    curve_path = os.path.join(RESULT_DIR, "cnn_loss_curves.png")
    plt.savefig(curve_path, dpi=300)
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
    plt.title("CNN Confusion Matrix", fontsize=18, pad=20)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULT_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"CNN confusion matrix saved to: {cm_path}")
    
    # Save training history
    logger.info("Saving training history")
    history_path = os.path.join(RESULT_DIR, "training_history.npz")
    np.savez(history_path, 
             cnn_train=hist_cnn["tr"], 
             cnn_val=hist_cnn["val"])
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