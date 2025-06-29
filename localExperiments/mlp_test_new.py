import os, sys, datetime, argparse, yaml
import numpy as np
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# ---------- 项目内依赖 ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mlp import MLP, TinyCNN  
from preprocessing.dataset import SHLWindowDataset   
from utils.utils import (get_logger, save_checkpoint,
                         AverageMeter, plot_confusion_matrix)

# ------------------------------ 超参数读取 ------------------------------ #
def parse_arguments():
    parser = argparse.ArgumentParser(description="MLP Training Script (new data pipeline)")
    parser.add_argument('--config', type=str,
                        default='localExperiments/model_param/mlp_cnn_params.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--model_name', type=str, default='MLP_test_v1',
                        help='配置文件中对应的 model_name')
    parser.add_argument('--data_root', type=str, default='processed_shl_balanced',
                        help='保存 *.npz 的目录，含 shl_train.npz / shl_val.npz')
    return parser.parse_args()

def load_config(cfg_path: str, model_name: str):
    with open(cfg_path, 'r') as f:
        cfg_all = yaml.safe_load(f)
    for m in cfg_all['model']:
        if m['model_name'] == model_name:
            return m
    raise ValueError(f"{model_name} 未在 {cfg_path} 中定义")

# ------------------------------ 数据加载 ------------------------------ #
def build_loader(npz_path: str | Path, batch_size: int,
                 shuffle: bool, num_workers: int = 0):
    ds = SHLWindowDataset(npz_path)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())

# ------------------------------ 评估函数 ------------------------------ #
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_meter, preds, trues = AverageMeter(), [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_meter.update(loss.item(), x.size(0))
            preds.extend(out.argmax(dim=1).cpu().tolist())
            trues.extend(y.cpu().tolist())
    acc = accuracy_score(trues, preds) * 100
    model.train()
    return loss_meter.avg, acc, (trues, preds)

# ------------------------------ 主训练流程 ------------------------------ #
def train_model(cfg, data_root: Path):
    # -------- 日志与目录 --------
    logs_dir = Path("localExperiments/logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("localExperiments/model_result/mlp/checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(filename=logs_dir/f"{cfg['model_name']}.log",
                        name=f"{cfg['model_name']}Logger",
                        overwrite=True, to_stdout=True)

    # -------- 数据 --------
    train_loader = build_loader(data_root/"shl_train.npz", cfg['batch_size'], shuffle=True)
    val_loader   = build_loader(data_root/"shl_val.npz",   cfg['batch_size'], shuffle=False)
    logger.info(f"Train/Val 样本数: {len(train_loader.dataset)}/{len(val_loader.dataset)}")

    # -------- 模型 & 优化器 --------
    model = MLP(input_size=cfg['input_size'], dropout=cfg['dropout'])
    # model = TinyCNN(in_ch=cfg['num_elements'], n_cls=8)  # 36 features, 8 classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = getattr(torch.nn, cfg['criterion'])()
    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), lr=cfg['lr'])

    best_acc = 0.0
    for epoch in range(cfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.float().to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))

        val_loss, val_acc, _ = evaluate(model, val_loader, device, criterion)
        logger.info(f"Epoch {epoch+1:03}/{cfg['epochs']} | "
                    f"TrainLoss {loss_meter.avg:.4f} | ValLoss {val_loss:.4f} | ValAcc {val_acc:.2f}%")

        # 保存权重
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        save_checkpoint(model, optimizer, epoch+1, loss_meter.avg, val_acc,
                        filename=ckpt_dir/f"{cfg['model_name']}_epoch{epoch+1}.pth",
                        is_best=is_best,
                        best_filename=ckpt_dir/f"{cfg['model_name']}_best.pth")

    logger.info(f"训练完成，最佳验证准确率 {best_acc:.2f}%")

# ------------------------------ 固定验证集评估 ------------------------------ #
def validate_best(cfg, data_root: Path):
    ckpt_dir = Path("localExperiments/model_result/mlp/checkpoints")
    best_path = ckpt_dir/f"{cfg['model_name']}_best.pth"
    assert best_path.exists(), f"未找到最佳模型 {best_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=cfg['input_size'], dropout=cfg['dropout'])
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    model.to(device).eval()

    val_loader = build_loader(data_root/"shl_val.npz", cfg['batch_size'], shuffle=False)
    criterion  = getattr(torch.nn, cfg['criterion'])()
    val_loss, val_acc, (trues, preds) = evaluate(model, val_loader, device, criterion)

    # -------- 混淆矩阵 --------
    class_names = [str(i+1) for i in range(8)]
    cm = confusion_matrix(trues, preds, labels=range(8))
    ax = plot_confusion_matrix(trues, preds, class_names,
                               normalize=True, fontsize=16)
    plt.title(f'Confusion Matrix – {cfg["model_name"]}\nValAcc {val_acc:.2f}%')
    out_png = Path("localExperiments/model_result/mlp_confusion_matrix_plots")
    out_png.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png/f"{cfg['model_name']}_val_cm.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------- 类别准确率日志 --------
    logger = get_logger(filename=Path("localExperiments/logs")/f"{cfg['model_name']}_val.log",
                        name=f"{cfg['model_name']}ValLogger",
                        overwrite=True, to_stdout=True)
    logger.info(f"ValLoss {val_loss:.4f}  ValAcc {val_acc:.2f}%")
    for i, c in enumerate(class_names):
        acc_cls = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        logger.info(f"Class {c}: {acc_cls:.4f}")

# ------------------------------ 可选 K-Fold ------------------------------ #
def train_model_cv(cfg, folds=5, seed=2025, data_root: Path = Path("processed_shl")):
    full_npz = data_root/"shl_train.npz"
    data = torch.load(full_npz) if full_npz.suffix == '.pt' else \
           dict(np.load(full_npz))
    X, y = torch.tensor(data['data']).float(), torch.tensor(data['labels']).long()
    ds_all = TensorDataset(X, y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    acc_list = []

    for f, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {f+1}/{folds} (train {len(tr_idx)} | val {len(val_idx)}) ===")
        tr_loader = DataLoader(Subset(ds_all, tr_idx), batch_size=cfg['batch_size'],
                               shuffle=True)
        val_loader = DataLoader(Subset(ds_all, val_idx), batch_size=cfg['batch_size'],
                                shuffle=False)

        model = MLP(input_size=cfg['input_size'], dropout=cfg['dropout'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        crit = getattr(torch.nn, cfg['criterion'])()
        opt  = getattr(torch.optim, cfg['optimizer'])(model.parameters(), lr=cfg['lr'])

        best = 0
        for ep in range(cfg['epochs']):
            model.train()
            for x, yb in tr_loader:
                x, yb = x.to(device), yb.to(device)
                opt.zero_grad(); loss = crit(model(x), yb); loss.backward(); opt.step()
            _, acc, _ = evaluate(model, val_loader, device, crit)
            best = max(best, acc)
        acc_list.append(best)
        print(f"Fold {f+1} best ValAcc={best:.2f}%")
    print(f"\n>> 5-Fold CV 结果: mean={sum(acc_list)/folds:.2f}%  "
          f"std={torch.tensor(acc_list).std():.2f}%")

# ============================== 入口 ===================================== #
if __name__ == "__main__":
    args  = parse_arguments()
    cfg   = load_config(args.config, args.model_name)
    root  = Path(args.data_root)

    train_model(cfg, root)           # 训练
    validate_best(cfg, root)         # 固定验证集评估
    # train_model_cv(cfg, folds=5)   # 若需交叉验证，取消注释
