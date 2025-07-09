"""
CNN backbone with Spiking Neurons for GeoMagnetic Dataset
"""

import os, random, time, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

from spikingjelly.activation_based import layer, neuron, functional, surrogate, encoding

# ---------- 0. Path & logger (reuse utils) ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

# ---------- 1. Reproducibility ----------
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Logging & directories ----------
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
logger.info(f"Running on: {DEVICE}, Seed: {SEED}")

# ---------- 3. Load and preprocess data ----------
DATA_PATH = "data/geoMag/geoMagDataset.npz"
data = np.load(DATA_PATH)
encoder = encoding.PoissonEncoder(step_mode='s')
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val     = data["X_test"] , data["y_test"]

assert X_train.shape[1] == 621, "Expected 621 flat features"
X_train = X_train.reshape(-1, 207, 3)  # (B, T, C)
X_val   = X_val.reshape(-1, 207, 3)

# ---------- 4. Dataset & DataLoader ----------
class SeqDataset(Dataset):
    """Tensor wrapper for sequence data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

batch_size = 16
dl_train = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
dl_val   = DataLoader(SeqDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
logger.info(f"Data ready. Train batches: {len(dl_train)}, Val batches: {len(dl_val)}")

# ---------- 5. Spiking Network Definition ----------

class SNN(nn.Module):
    """1-D CNN backbone with IF neurons."""
    def __init__(self, input_channels=3, num_classes=3, time_steps=16):
        super().__init__()
        self.T = time_steps
        self.conv1 = layer.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.sn1   = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dp1   = nn.Dropout(0.2)

        self.conv2 = layer.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.sn2   = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dp2   = nn.Dropout(0.2)

        self.conv3 = layer.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(32)
        self.sn3   = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.dp3   = nn.Dropout(0.2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        x shape: (B, T_len, C_in)  ->  transpose to (B, C_in, T_len)
        Simulate for self.T steps; reuse same analog input each step (rate coding).
        """
        x = x.transpose(1, 2)
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

        return out_spk_sum / self.T          # vote-count

model = SNN().to(DEVICE)
logger.info(f"SNN parameters: {sum(p.numel() for p in model.parameters())}")

# ---------- 6. Training Function ----------
def train(net, num_epochs=200, lr=1e-3, T=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    history, best_val = {"tr": [], "val": []}, float("inf")
    start_time = time.time()

    logger.info(f"Start SNN training for {num_epochs} epochs")
    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        net.train()
        tr_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            functional.reset_net(net)          # clear membranes
            out_fr = 0.0                       # reset output accumulator
            for t in range(T):
                encoded_xb = encoder(xb)  # Poisson encoding
                out_fr += net(encoded_xb)  # forward pass
            logits = out_fr / T  # average over time steps
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(dl_train.dataset)

        # ---- val ----
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                functional.reset_net(net)
                out_fr = 0.0  # reset output accumulator
                for t in range(T):
                    encoded_xb = encoder(xb)
                    out_fr = net(encoded_xb)  # forward pass
                # average over time steps
                logits = out_fr / T
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(dl_val.dataset)

        history["tr"].append(tr_loss)
        history["val"].append(val_loss)

        # checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"snn_epoch_{epoch:03d}.pth")
        is_best   = val_loss < best_val
        if is_best:
            best_val = val_loss
        save_checkpoint(
            model=net, optimizer=optimizer, epoch=epoch, loss=val_loss,
            filename=ckpt_path, is_best=is_best,
            best_filename=os.path.join(CHECKPOINT_DIR, "snn_best.pth")
        )
        if epoch % 10 == 0 or epoch <= 5:
            logger.info(f"SNN | Epoch {epoch:03d}  Train {tr_loss:.4f}  Val {val_loss:.4f}")
        if is_best:
            logger.info(f"SNN | New best at epoch {epoch}  Val {val_loss:.4f}")

    logger.info(f"SNN training finished in {time.time()-start_time:.2f} s")
    torch.save(net.state_dict(), os.path.join(RESULT_DIR, "snn_model.pth"))
    return history

# ---------- 7. Evaluation ----------
def evaluate(net, T=100):
    net.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(DEVICE)
            functional.reset_net(net)
            out_fr = 0.0  # reset output accumulator
            for t in range(T):  # simulate for 100 time steps
                encoded_xb = encoder(xb)
                out_fr = net(encoded_xb)  # forward pass
            logits = out_fr / T # average over time steps
            y_pred.append(logits.cpu().argmax(1))
            y_true.append(yb)
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()

    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc  = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    f1  = f1_score(y_true, y_pred, average="weighted",     zero_division=0)
    logger.info(f"SNN Result | Acc:{acc:.4f}  Prec:{pr:.4f}  Recall:{rc:.4f}  F1:{f1:.4f}")
    logger.info("Classification report:\n" +
                classification_report(y_true, y_pred, digits=4))
    return acc, pr, rc, f1

# ---------- 8. Main ----------
if __name__ == "__main__":
    logger.info("="*60)
    hist = train(model, num_epochs=200)
    logger.info("="*60)
    evaluate(model)
