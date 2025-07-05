import os, sys, yaml, argparse, datetime, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import get_logger, AverageMeter, save_checkpoint, plot_confusion_matrix, select_certain_classes  
from sklearn.metrics import confusion_matrix, f1_score
import psutil

# ----------------------------------------------------------------------
# 1. Model Definition
# ----------------------------------------------------------------------
class LSTM_MLP(nn.Module):
    """
    input:  (B, C=20, T=window_size) 
    output:  (B, 8)
    """
    def __init__(self,
                 num_elements: int = 20,
                 window_size: int = 300,
                 hidden_dim:  int = 128,
                 num_classes: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM — batch_first 使用 (B, T, C) 形状
        self.lstm = nn.LSTM(input_size=num_elements,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        # MLP 5 层: 128-256-512-1024-8
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, num_classes)   
        )

    def forward(self, x):
        # x: (B, C, T)  →  (B, T, C)
        # x = x.permute(0, 2, 1).contiguous()
        out, _ = self.lstm(x)              # out: (B, T, H)

        last_out = out[:, -1, :]           # (B, H)
        logits = self.mlp(last_out)        # (B, num_classes)
        return logits

# ----------------------------------------------------------------------
# 2. YAML
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LSTM-MLP Training Script")
    p.add_argument("--config", type=str,
                   default="localExperiments/model_param/lstm_mlp_params.yaml")
    p.add_argument("--model_name", type=str,
                   default="LSTM_MLP_test_v2")
    return p.parse_args()

def load_config(cfg_path, model_name):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    for m in cfg["model"]:
        if m["model_name"] == model_name:
            return m
    raise ValueError(f"未找到模型 {model_name} 于 {cfg_path}")

# ----------------------------------------------------------------------
# 3. evaluate function
# ----------------------------------------------------------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_meter, correct, total = AverageMeter(), 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).long()
            logits = model(x)
            loss = criterion(logits, y)
            loss_meter.update(loss.item(), x.size(0))
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    model.train()
    return loss_meter.avg, 100.*correct/total

# ----------------------------------------------------------------------
# 4. train main
# ----------------------------------------------------------------------
def train_model(cfg):
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    logger = get_logger(filename=os.path.join(
                            cfg["log_dir"], f"{cfg['model_name']}.log"),
                        name=f"{cfg['model_name']}_Logger",
                        overwrite=True, to_stdout=True)

    train_data = select_certain_classes('data/SHL_2018/all_data_train_0.8_window_450_overlap_0.0.npz', 
                                        selected_classes=["mag_x", "mag_y", "mag_z"])
    val_data = select_certain_classes('data/SHL_2018/all_data_test_0.8_window_450_overlap_0.0.npz', 
                                      selected_classes=["mag_x", "mag_y", "mag_z"])

    train_x, train_y = train_data['x'].astype(np.float32), train_data['y']
    val_x, val_y = val_data['x'].astype(np.float32), val_data['y']

    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTM_MLP(num_elements=cfg["num_elements"],
                      window_size=cfg["window_size"],
                      hidden_dim=128,
                      num_classes=cfg["num_classes"],
                      dropout=cfg["dropout"]).to(device)

    criterion = getattr(nn, cfg["criterion"])()
    optimizer = getattr(torch.optim, cfg["optimizer"])(model.parameters(),
                                                      lr=cfg["lr"])

    logger.info(model)

    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        loss_meter = AverageMeter()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device).float(), y.to(device).long()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))

            if step % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{cfg['epochs']} "
                            f"Step {step:<4d} Loss {loss.item():.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        logger.info(f"[Epoch {epoch+1}] TrainLoss {loss_meter.avg:.4f} | "
                    f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.2f}%")

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        save_checkpoint(model, optimizer, epoch+1, loss_meter.avg, val_acc,
                        filename=os.path.join(cfg["ckpt_dir"],
                                              f"{cfg['model_name']}_epoch{epoch+1}.pth"),
                        is_best=is_best,
                        best_filename=os.path.join(cfg["ckpt_dir"],
                                                   f"{cfg['model_name']}_best.pth"))

    logger.info(f"Training finished, best validation accuracy: {best_acc:.2f}%")

# ----------------------------------------------------------------------
# 5. result_validation
# ----------------------------------------------------------------------
def result_validation(config, fold=None):
    """
    Evaluate the best model on the fixed validation set (shl_validation.npz)
    fold: Specify the fold number to load (1-5), if None, automatically select the best fold
    """
    logs_dir = "localExperiments/logs"
    logger = get_logger(
        filename=os.path.join(logs_dir, f"{config['model_name']}_val.log"),
        name=f"{config['model_name']}ValLogger",
        level="DEBUG", overwrite=True, to_stdout=True
    )

    # ---------- 1. Load Model ----------
    ckpt_dir = "localExperiments/model_result/lstm_mlp/checkpoints"

    if fold is not None:
        # Load specified fold model
        best_path = os.path.join(ckpt_dir, f"{config['model_name']}_fold{fold}_best.pth")
        assert os.path.exists(best_path), f"找不到模型 {best_path}"
    else:
        # Choose the best fold automatically
        best_acc = 0.0
        best_path = os.path.join(ckpt_dir, f"{config['model_name']}_best.pth")
        
        assert best_path is not None, f"Found no valid model checkpoints for {config['model_name']}"
        logger.info(f"Automatically selected best fold: fold{fold} (Validation accuracy: {best_acc:.2f}%)")

    # Define device before using it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model  = LSTM_MLP(num_elements=config["num_elements"],
                      window_size=config["window_size"],
                      hidden_dim=128,
                      num_classes=config["num_classes"],
                      dropout=config["dropout"]).to(device)
    
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    logger.info(f"Loaded fold{fold} model: ValAcc in training ={checkpoint['accuracy']:.2f}%")

    # ---------- 2. Model Size and System Information ----------
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model file size
    model_file_size = os.path.getsize(best_path) / (1024 * 1024)  # Convert to MB
    
    # Get system information
    system_info = {
        'CPU': psutil.cpu_count(),
        'Memory': psutil.virtual_memory().total / (1024**3),  # Convert to GB
        'Available Memory': psutil.virtual_memory().available / (1024**3),  # Convert to GB
    }
    
    # GPU information if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        system_info['GPU'] = torch.cuda.get_device_name(0)
        system_info['GPU Memory'] = gpu_memory
    
    logger.info("=" * 50)
    logger.info("MODEL AND SYSTEM INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Model File Size: {model_file_size:.2f} MB")
    logger.info(f"CPU Cores: {system_info['CPU']}")
    logger.info(f"Total Memory: {system_info['Memory']:.2f} GB")
    logger.info(f"Available Memory: {system_info['Available Memory']:.2f} GB")
    if 'GPU' in system_info:
        logger.info(f"GPU: {system_info['GPU']}")
        logger.info(f"GPU Memory: {system_info['GPU Memory']:.2f} GB")
    logger.info("=" * 50)

    val_data = select_certain_classes('data/SHL_2018/all_data_test_0.8_window_450_overlap_0.0.npz', 
                                      selected_classes=["mag_x", "mag_y", "mag_z"])
    
    val_x, val_y = val_data['x'].astype(np.float32), val_data['y']
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    logger.info(f"Validation data loaded: {len(val_loader)} batches")

    # ---------- Evaluation ----------
    criterion = getattr(torch.nn, config['criterion'])()
    val_loss, val_acc = evaluate(model, val_loader, device, criterion)
    logger.info(f"Validation Loss={val_loss:.4f}  Accuracy={val_acc:.2f}%")

    # ---------- Detailed evaluation with F1 score and inference time ----------
    all_preds, all_true = [], []
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            batch_inference_time = (end_time - start_time) / inputs.size(0)  # Per sample
            inference_times.extend([batch_inference_time] * inputs.size(0))
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(targets.cpu().numpy())

    # Calculate F1 score
    f1_macro = f1_score(all_true, all_preds, average='macro')
    f1_weighted = f1_score(all_true, all_preds, average='weighted')
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds
    
    logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    logger.info(f"Average Inference Time: {avg_inference_time:.4f} ms per sample")


    class_names = [str(i) for i in range(8)]  

    cm = confusion_matrix(all_true, all_preds, labels=range(8)) 
    ax = plot_confusion_matrix(
        test_y=all_true, 
        pred_y=all_preds,
        class_names=class_names,
        normalize=True, 
        fontsize=18
    )
    
    plt.title(f'Confusion Matrix – {config["model_name"]}\nValAcc {val_acc:.2f}% | F1 {f1_macro:.4f}')
    results_dir = "localExperiments/model_result/cnn_confusion_matrix_plots"
    os.makedirs(results_dir, exist_ok=True)
    fig_path = os.path.join(results_dir, f"{config['model_name']}_val_cm.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Confusion matrix saved to {fig_path}")

    # ---------- 5. Class-wise accuracy ----------
    logger.info("Class-wise accuracy:")
    for i, cls in enumerate(class_names):
        cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        logger.info(f"Class {cls}: {cls_acc:.4f}")
    
    # ---------- 6. Summary ----------
    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Model File Size: {model_file_size:.2f} MB")
    logger.info(f"Validation Accuracy: {val_acc:.2f}%")
    logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    logger.info(f"Average Inference Time: {avg_inference_time:.4f} ms per sample")
    logger.info("=" * 50)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    args  = parse_args()
    cfg   = load_config(args.config, args.model_name)
    # train_model(cfg)
    result_validation(cfg)
