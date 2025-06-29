import sys
import os
import datetime
import yaml
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.mlp import MLP
from preprocessing.getDataset import get_npz_dataloader, get_mag_dataloader
from utils.utils import get_logger, save_checkpoint, AverageMeter, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, Subset, DataLoader

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="MLP Training Script")
    parser.add_argument('--config', type=str, default='localExperiments/model_param/mlp_cnn_params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_name', type=str, default='TinyCNN_test_v1',
                        help='Model name as specified in the configuration file')
    return parser.parse_args()

def load_config(config_path, model_name):
    """
    Load model configuration from yaml file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find the specific model configuration
    for model_config in config['model']:
        if model_config['model_name'] == model_name:
            return model_config
    
    raise ValueError(f"Model configuration for {model_name} not found in {config_path}")

def evaluate(model, data_loader, device, criterion):
    """
    Evaluate model performance on given data loader
    """
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)

            # targets = targets - 1  # Adjust to CrossEntropy's 0-based indexing

            outputs = model(inputs)
            
            loss_value = criterion(outputs, targets)
            loss_meter.update(loss_value.item(), inputs.size(0))
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    model.train()
    return loss_meter.avg, accuracy

def train_model(config):
    """
    Train model according to provided configuration
    Use shl_train.npz for training and shl_validation.npz for validation
    """
    # Setup directories
    logs_dir = "localExperiments/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    checkpoints_dir = "localExperiments/model_result/mlp/checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Initialize logger
    logger = get_logger(
        filename=os.path.join(logs_dir, f"{config['model_name']}.log"),
        name=f"{config['model_name']}Logger",
        level="DEBUG",
        overwrite=True,
        to_stdout=True
    )

    logger.info(f"Starting training for {config['model_name']}")
    logger.info(f"Configuration: {config}")

    # Get data loaders
    if config['dataLoader'] == 'get_npz_dataloader':
        data_loader_func = get_npz_dataloader
    elif config['dataLoader'] == 'get_mag_dataloader':
        data_loader_func = get_mag_dataloader
    else:
        raise ValueError(f"Unknown dataLoader: {config['dataLoader']}")
    
    train_loader = data_loader_func(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_train.npz",
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = data_loader_func(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_validation.npz",
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Train loaded: {len(train_loader)} batches")
    logger.info(f"Validation loaded: {len(val_loader)} batches")

    # Initialize model
    model = MLP(input_size=config['input_size'], dropout=config['dropout'])
    logger.info(f"Model architecture:\n{model}")
    
    # Setup device, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    criterion = getattr(torch.nn, config['criterion'])()
    optimizer_class = getattr(torch.optim, config['optimizer'])
    optimizer = optimizer_class(model.parameters(), lr=config['lr'])
    
    # Training loop
    logger.info(f"Train dataset has {len(train_loader)} batches with batch size {train_loader.batch_size}")
    
    best_accuracy = 0.0
    for epoch in range(config['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        model.train()
        train_loss_meter = AverageMeter()
        batch_count = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            batch_count += 1
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)

            # targets = targets - 1 # Adjust to CrossEntropy's 0-based indexing

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            
            train_loss_meter.update(loss_value.item(), inputs.size(0))
            
            if i % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Step {i}, Batch count: {batch_count}, Loss: {loss_value.item():.4f}")
        
        logger.info(f"Epoch {epoch+1} had {batch_count} batches")
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, val_loader, device, criterion)
        logger.info(f"Epoch {epoch+1} completed. Train Loss: {train_loss_meter.avg:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save checkpoint
        is_best = val_accuracy > best_accuracy
        best_accuracy = max(val_accuracy, best_accuracy)
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch+1,
            loss=train_loss_meter.avg,
            accuracy=val_accuracy,
            filename=os.path.join(checkpoints_dir, f"{config['model_name']}_epoch{epoch+1}.pth"),
            is_best=is_best,
            best_filename=os.path.join(checkpoints_dir, f"{config['model_name']}_best.pth")
        )
    
    logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
    
    # Write summary
    summary_path = os.path.join(logs_dir, f"{config['model_name']}_summary.txt")
    model_name = config['model_name']
    best_model_path = os.path.join(checkpoints_dir, f"{model_name}_best.pth")
    
    with open(summary_path, "w") as f:
        f.write(f"Train completed at: {datetime.datetime.now()}\n")
        f.write(f"Best validation accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Model saved at: {best_model_path}\n")
        f.write(f"Configuration: {config}\n")

def train_model_cv(config, folds=5, seed=2025):
    """
    只使用 shl_train.npz，对其做 K-Fold 交叉验证。
    """
    # --------------- 设置数据加载器 ----------------
    if config['dataLoader'] == 'get_npz_dataloader':
        data_loader_func = get_npz_dataloader
    elif config['dataLoader'] == 'get_mag_dataloader':
        data_loader_func = get_mag_dataloader
    else:
        raise ValueError(f"Unknown dataLoader: {config['dataLoader']}")

    # --------------- 加载全部数据 ----------------
    train_npz_path = "/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_train.npz"
    data = np.load(train_npz_path)
    X_all = torch.from_numpy(data['data']).float()   # (N, C, W)
    y_all = torch.from_numpy(data['labels']).long()    # (N,)
    
    full_ds = TensorDataset(X_all, y_all)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"\n===== Fold {fold+1}/{folds}  | train={len(train_idx)}  val={len(val_idx)} =====")

        # --------------- 2. DataLoader ----------------
        # 使用与train_model相同的数据加载器，但需要使用Subset来划分数据
        full_loader = data_loader_func(
            npz_path=train_npz_path,
            batch_size=config['batch_size'],
            shuffle=False,  # 不需要打乱，因为我们会手动选择索引
            num_workers=0
        )
        
        # 创建训练集和验证集的子集
        train_subset = Subset(full_loader.dataset, train_idx)
        val_subset = Subset(full_loader.dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # --------------- Logger ----------------
        logs_dir = "localExperiments/logs"
        os.makedirs(logs_dir, exist_ok=True)
        ckpt_dir = "localExperiments/model_result/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        fold_tag = f"{config['model_name']}_fold{fold+1}"
        logger = get_logger(
            filename=os.path.join(logs_dir, f"{fold_tag}.log"),
            name=f"{fold_tag}Logger",
            level="DEBUG", overwrite=True, to_stdout=True
        )

        # --------------- Model / Optimizer ----------------
        model = MLP(input_size=config['input_size'], dropout=config['dropout'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = getattr(torch.nn, config['criterion'])()
        optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), lr=config['lr'])

        best_acc = 0.0
        for epoch in range(config['epochs']):
            model.train()
            loss_meter = AverageMeter()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item(), inputs.size(0))

            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            logger.info(f"Fold {fold+1} | Epoch {epoch+1}/{config['epochs']} "
                        f"TrainLoss {loss_meter.avg:.4f}  ValAcc {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch+1, loss_meter.avg, val_acc,
                    filename=os.path.join(ckpt_dir, f"{fold_tag}_epoch{epoch+1}.pth"),
                    is_best=True,
                    best_filename=os.path.join(ckpt_dir, f"{fold_tag}_best.pth")
                )

        logger.info(f"=== Fold {fold+1} finished. Best ValAcc={best_acc:.2f}% ===")
        fold_metrics.append(best_acc)

    mean_acc = np.mean(fold_metrics)
    std_acc  = np.std (fold_metrics)
    print(f"\n##### 5-Fold CV finished: mean={mean_acc:.2f}%  std={std_acc:.2f}% #####")



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
    ckpt_dir = "localExperiments/model_result/mlp/checkpoints"
    
    if fold is not None:
        # Load specified fold model
        best_path = os.path.join(ckpt_dir, f"{config['model_name']}_fold{fold}_best.pth")
        assert os.path.exists(best_path), f"找不到模型 {best_path}"
    else:
        # Choose the best fold automatically
        best_acc = 0.0
        best_path = os.path.join(ckpt_dir, f"{config['model_name']}_best.pth")
        # 省略部分代码...
        
        assert best_path is not None, f"Found no valid model checkpoints for {config['model_name']}"
        logger.info(f"Automatically selected best fold: fold{fold} (Validation accuracy: {best_acc:.2f}%)")

    model = MLP(input_size=config['input_size'], dropout=config['dropout'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    logger.info(f"Loaded fold{fold} model: ValAcc in training ={checkpoint['accuracy']:.2f}%")

    # ---------- 2. 使用一致的数据加载器 ----------
    if config['dataLoader'] == 'get_npz_dataloader':
        data_loader_func = get_npz_dataloader
    elif config['dataLoader'] == 'get_mag_dataloader':
        data_loader_func = get_mag_dataloader
    else:
        raise ValueError(f"Unknown dataLoader: {config['dataLoader']}")
    
    val_loader = data_loader_func(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_validation.npz",
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    logger.info(f"Validation data loaded: {len(val_loader)} batches")

    # ---------- Evaluation ----------
    criterion = getattr(torch.nn, config['criterion'])()
    val_loss, val_acc = evaluate(model, val_loader, device, criterion)
    logger.info(f"Validation Loss={val_loss:.4f}  Accuracy={val_acc:.2f}%")

    # ---------- Confusion Matrix ----------
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 调整标签与evaluate函数一致
            # targets_adjusted = targets - 1  # 调整到0-7范围
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(targets.cpu().numpy())

    # 确保类别标签是0-7范围内的整数
    class_names = [str(i+1) for i in range(8)]  # 使用1-8作为显示的类别名称
    
    # 调用混淆矩阵绘制函数
    cm = confusion_matrix(all_true, all_preds, labels=range(8))  # 明确指定标签范围
    ax = plot_confusion_matrix(
        test_y=all_true, 
        pred_y=all_preds,
        class_names=class_names,
        normalize=True, 
        fontsize=18
    )
    
    plt.title(f'Confusion Matrix – {config["model_name"]}\nValAcc {val_acc:.2f}%')
    results_dir = "localExperiments/model_result/mlp_confusion_matrix_plots"
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


if __name__ == "__main__":
    args = parse_arguments()
    cfg = load_config(args.config, args.model_name)
    
    train_model(cfg)
    # train_model_cv(cfg, folds=5, seed=2025)   

    result_validation(cfg, fold=None)


