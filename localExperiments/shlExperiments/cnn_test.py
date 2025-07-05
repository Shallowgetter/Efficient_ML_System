import sys
import os
import datetime
import yaml
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.cnn import CNN, CNNGAP
from torch.utils.data import DataLoader
from utils.utils import get_logger, save_checkpoint, AverageMeter, plot_confusion_matrix, select_certain_classes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import random
import time
import psutil

from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, Subset, DataLoader

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="CNN Training Script")
    parser.add_argument('--config', type=str, default='localExperiments/model_param/mlp_cnn_params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_name', type=str, default='CNN_test_v4',
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

    checkpoints_dir = "localExperiments/model_result/cnn/checkpoints"
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

    train_data = select_certain_classes('data/SHL_2018/all_data_train_0.8_window_450_overlap_0.0.npz', 
                                        selected_classes=["mag_x", "mag_y", "mag_z"])
    val_data = select_certain_classes('data/SHL_2018/all_data_test_0.8_window_450_overlap_0.0.npz', 
                                      selected_classes=["mag_x", "mag_y", "mag_z"])

    train_x, train_y = train_data['x'].astype(np.float32), train_data['y']
    val_x, val_y = val_data['x'].astype(np.float32), val_data['y']

    # Dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    logger.info(f"Train loaded: {len(train_loader)} batches")
    logger.info(f"Validation loaded: {len(val_loader)} batches")

    # Initialize model - 修改这部分使其完全参数化
    model_type = config.get('model_type', 'CNN')  # 默认使用CNN
    
    if model_type == 'CNN':
        model = CNN(
            num_elements=config['num_elements'], 
            window_size=config['window_size'], 
            num_classes=config['num_classes'], 
            dropout=config.get('dropout', 0.5)  # 使用get方法提供默认值
        )
    elif model_type == 'CNNGAP':
        model = CNNGAP(
            num_elements=config['num_elements'], 
            window_size=config['window_size'], 
            num_classes=config['num_classes'], 
            dropout=config.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
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
    ckpt_dir = "localExperiments/model_result/cnn/checkpoints"

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

    model_type = config.get('model_type', 'CNN')  
    
    if model_type == 'CNN':
        model = CNN(
            num_elements=config['num_elements'], 
            window_size=config['window_size'], 
            num_classes=config['num_classes'], 
            dropout=config.get('dropout', 0.5)
        )
    elif model_type == 'CNNGAP':
        model = CNNGAP(
            num_elements=config['num_elements'], 
            window_size=config['window_size'], 
            num_classes=config['num_classes'], 
            dropout=config.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # ---------- 3. 使用一致的数据加载器 ----------
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

    # 确保类别标签是0-7范围内的整数
    class_names = [str(i) for i in range(8)]  # 使用0-7作为显示的类别名称

    # 调用混淆矩阵绘制函数
    cm = confusion_matrix(all_true, all_preds, labels=range(8))  # 明确指定标签范围
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


if __name__ == "__main__":
    args = parse_arguments()
    cfg = load_config(args.config, args.model_name)
    
    train_model(cfg)   

    result_validation(cfg, fold=None)