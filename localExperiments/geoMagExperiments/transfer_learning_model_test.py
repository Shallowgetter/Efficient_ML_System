# transfer_learning_model_test.py
# Transfer-learning baselines: DenseNet201/169/121, MobileNet(v3,v2), VGG16/19
# ---------------------------------------------------------------------------

import os, sys, time, random
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ---------- import utilities (same as seq script) --------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from utils.utils import get_logger, save_checkpoint, plot_confusion_matrix

# ---------------- 1. logging / dirs / seed ---------------------------------
RESULT_DIR     = "localExperiments/geoMagExperiments/model_result/TLModelResult"
CHECKPOINT_DIR = "localExperiments/geoMagExperiments/model_result/transfer_model_checkPoints"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger = get_logger(
    filename=os.path.join(RESULT_DIR, "tl_experiment.log"),
    name="TL_Experiment",
    overwrite=True,
    to_stdout=True,
    level="INFO"
)

SEED = 10
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ---------------- 2. load image dataset ------------------------------------
IMG_NPZ = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/geoMag/geoMagDataset_img.npz"    # produced by updated getDataset.py
logger.info(f"Loading image dataset: {IMG_NPZ}")
img_data = np.load(IMG_NPZ)
X_train_img = img_data["X_train_img"]   # (621, 216,216,3) uint8
y_train     = img_data["y_train"]
X_test_img  = img_data["X_test_img"]
y_test      = img_data["y_test"]
class_names = ["Light", "Medium", "Heavy"]
logger.info(f"Dataset shapes  Train:{X_train_img.shape}  Test:{X_test_img.shape}")

# ------------- 3. Dataset / DataLoader definitions -------------------------
class ImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tf = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.imgs[idx]      # uint8 H,W,C
        if self.tf: img = self.tf(img)
        return img, self.labels[idx]

mean = [0.5, 0.5, 0.5]; std = [0.5, 0.5, 0.5]
train_tf = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(15),
    T.RandomResizedCrop(224, scale=(0.9, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std)
])
test_tf = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std)
])

batch_size = 16
dl_train = DataLoader(ImageDataset(X_train_img, y_train, train_tf),
                      batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
dl_test  = DataLoader(ImageDataset(X_test_img,  y_test,  test_tf),
                      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
logger.info(f"DataLoaders ready | Train batches:{len(dl_train)} Test batches:{len(dl_test)}")

# ------------- 4. helper to build TL model ---------------------------------
def build_backbone(name: str):
    name = name.lower()
    if name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    elif name == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    elif name == "mobilenet":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        # MobileNetV3 的 classifier 结构: [Linear(960, 1280), Hardswish(), Dropout(0.2), Linear(1280, 1000)]
        num_ftrs = model.classifier[0].in_features  # 960
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    elif name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # MobileNetV2 的 classifier 结构: [Dropout(0.2), Linear(1280, 1000)]
        num_ftrs = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[0].in_features  # 25088
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    elif name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    # freeze convolutional base
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():      # only head is trainable
        p.requires_grad = True
    return model.to(DEVICE)

# ------------- 5. train / evaluate functions -------------------------------
def train(model, name, epochs=50):
    logger.info(f"--- Training {name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(),
                                 lr=1e-2, weight_decay=1e-4)
    hist = {"tr": [], "val": []}
    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            running += loss.item() * xb.size(0)
        tr_loss = running / len(dl_train.dataset)

        model.eval(); running = 0.0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                running += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = running / len(dl_test.dataset)
        hist["tr"].append(tr_loss); hist["val"].append(val_loss)

        if ep % 10 == 0 or ep <= 5:
            logger.info(f"{name}  Epoch {ep:02d}  Train {tr_loss:.4f}  Val {val_loss:.4f}")

        chk = os.path.join(CHECKPOINT_DIR, f"{name}_ep{ep:02d}.pth")
        save_checkpoint(model, optimizer, ep, val_loss, chk)
    # save final
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, f"{name}.pth"))
    return hist

@torch.no_grad()
def evaluate(model, name):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in dl_test:
        xb = xb.to(DEVICE)
        all_pred.append(model(xb).cpu().argmax(1))
        all_true.append(yb)
    y_pred = torch.cat(all_pred).numpy(); y_true = torch.cat(all_true).numpy()
    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc  = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    f1  = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    rep = classification_report(y_true, y_pred, digits=4)
    logger.info(f"{name} metrics  Acc:{acc:.4f}  Prec:{pr:.4f}  Rec:{rc:.4f}  F1:{f1:.4f}")
    with open(os.path.join(RESULT_DIR, f"{name}_report.txt"), "w") as f:
        f.write(rep)
    # confusion matrix figure
    fig_path = os.path.join(RESULT_DIR, f"cm_{name}.png")
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=True,
                          fontsize=14, axis=1, vmin=0, vmax=1).get_figure().savefig(fig_path, dpi=300)
    return acc, pr, rc, f1, rep

# ------------- 6. main loop over backbones ---------------------------------
BACKBONES = ["densenet201", "densenet169", "densenet121",
             "mobilenet", "mobilenetv2", "vgg16", "vgg19"]

all_metrics = {}
for bb in BACKBONES:
    model = build_backbone(bb)
    hist  = train(model, bb.upper())
    acc, pr, rc, f1, _ = evaluate(model, bb.upper())
    all_metrics[bb] = (acc, pr, rc, f1)
    # save loss curve
    plt.figure(figsize=(8,4))
    plt.plot(hist["tr"], label="Train")
    plt.plot(hist["val"], label="Val")
    plt.title(f"{bb.upper()} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{bb}_loss.png"), dpi=300); plt.close()

# ------------- 7. summary ---------------------------------------------------
with open(os.path.join(RESULT_DIR, "summary_metrics.txt"), "w") as f:
    for k,(a,p,r,f1) in all_metrics.items():
        line = f"{k.upper():12s}  Acc:{a:.4f}  Prec:{p:.4f}  Rec:{r:.4f}  F1:{f1:.4f}"
        logger.info(line); f.write(line+"\n")

logger.info("Transfer-learning experiment finished. See RESULT_DIR for outputs.")
