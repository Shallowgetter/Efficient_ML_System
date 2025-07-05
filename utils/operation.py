import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from .utils import plot_confusion_matrix, get_logger, AverageMeter, load_checkpoint, save_checkpoint
import psutil

logger = get_logger("logs/training.log", name="TRAINING", level="INFO")

def ann_train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    logger.info(f"Starting training {model} model for {num_batches} batches...")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
            avg_loss = total_loss / (batch_idx + 1)
            logger.info(f"Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

    avg_loss = total_loss / num_batches
    logger.info(f"Training complete | Avg Loss: {avg_loss:.4f}")


def ann_evaluate(model: nn.Module, eval_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = len(eval_loader)

    logger.info(f"Starting evaluation {model} model for {num_batches} batches...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

    avg_loss = total_loss / num_batches
    logger.info(f"Evaluation complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_name: str = "Model",
    save_dir: str = "checkpoints",
    log_interval: int = 100,
    save_best_only: bool = False
) -> dict:
    """
    Train and validate model with comprehensive logging and checkpointing.
    
    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        DataLoader for training dataset.
    val_loader : DataLoader
        DataLoader for validation dataset.
    criterion : nn.Module
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for training.
    device : torch.device
        Device to run training on (CPU or GPU).
    epochs : int
        Number of training epochs.
    model_name : str, optional
        Name of the model for logging and saving checkpoints.
    save_dir : str, optional
        Directory to save model checkpoints.
    log_interval : int, optional
        Interval for logging training progress (in batches).
    save_best_only : bool, optional
        Whether to save only the best model or all epoch checkpoints.
    
    Returns
    -------
    dict
        Dictionary containing training history and best model info.
    """
    # Create training logger
    train_logger = get_logger(
        filename=f"logs/{model_name}_training.log",
        name=f"{model_name}_TRAIN",
        level="INFO",
        overwrite=True
    )
    
    train_logger.info("=" * 60)
    train_logger.info("STARTING MODEL TRAINING")
    train_logger.info("=" * 60)
    
    # Setup directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Log training configuration
    train_logger.info(f"Model: {model_name}")
    train_logger.info(f"Device: {device}")
    train_logger.info(f"Epochs: {epochs}")
    train_logger.info(f"Train batches: {len(train_loader)}")
    train_logger.info(f"Validation batches: {len(val_loader)}")
    train_logger.info(f"Batch size: {train_loader.batch_size}")
    train_logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    train_logger.info(f"Optimizer: {type(optimizer).__name__}")
    train_logger.info(f"Criterion: {type(criterion).__name__}")
    
    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_logger.info(f"Total Parameters: {total_params:,}")
    train_logger.info(f"Trainable Parameters: {trainable_params:,}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'epochs': [],
        'best_epoch': 0,
        'best_accuracy': 0.0
    }
    
    model.to(device)
    best_accuracy = 0.0
    
    train_logger.info("=" * 60)
    train_logger.info("TRAINING STARTED")
    train_logger.info("=" * 60)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase - use our ann_train function with modifications
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        train_logger.info(f"Starting training phase...")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Handle different model output formats
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Log training progress
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                train_logger.info(f"  Batch {batch_idx + 1}/{num_batches} | "
                                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
        
        avg_train_loss = total_loss / num_batches
        train_logger.info(f"Training phase complete | Avg Loss: {avg_train_loss:.4f}")
        
        # Validation phase - use our ann_evaluate function with modifications
        model.eval()
        val_total_loss = 0.0
        correct = 0
        total = 0
        
        train_logger.info(f"Starting validation phase...")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Handle different model output formats
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                
                loss = criterion(outputs, targets)
                val_total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_total_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        train_logger.info(f"Validation phase complete | Avg Loss: {avg_val_loss:.4f}")
        train_logger.info(f"Epoch {epoch+1} Summary:")
        train_logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        train_logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        train_logger.info(f"  Val Accuracy: {val_accuracy:.2f}%")
        train_logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['epochs'].append(epoch + 1)
        
        # Save checkpoint
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy
            history['best_epoch'] = epoch + 1
            history['best_accuracy'] = best_accuracy
            train_logger.info(f"New best model! Validation accuracy: {best_accuracy:.2f}%")
        
        checkpoint_filename = os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pth")
        best_filename = os.path.join(save_dir, f"{model_name}_best.pth")
        
        if save_best_only:
            if is_best:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch+1,
                    loss=avg_train_loss,
                    accuracy=val_accuracy,
                    filename=best_filename,
                    is_best=False,  # Don't duplicate save
                    best_filename=best_filename
                )
                train_logger.info(f"Best model saved to {best_filename}")
        else:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch+1,
                loss=avg_train_loss,
                accuracy=val_accuracy,
                filename=checkpoint_filename,
                is_best=is_best,
                best_filename=best_filename
            )
            train_logger.info(f"Checkpoint saved to {checkpoint_filename}")
        
        train_logger.info("-" * 60)
    
    # Training completion summary
    train_logger.info("=" * 60)
    train_logger.info("TRAINING COMPLETED")
    train_logger.info("=" * 60)
    train_logger.info(f"Best Epoch: {history['best_epoch']}")
    train_logger.info(f"Best Validation Accuracy: {history['best_accuracy']:.2f}%")
    train_logger.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    train_logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    train_logger.info(f"Total Parameters: {total_params:,}")
    train_logger.info(f"Model saved to: {os.path.join(save_dir, f'{model_name}_best.pth')}")
    
    # Save training history
    history_path = os.path.join(save_dir, f"{model_name}_history.txt")
    with open(history_path, "w") as f:
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Best epoch: {history['best_epoch']}\n")
        f.write(f"Best validation accuracy: {history['best_accuracy']:.2f}%\n")
        f.write(f"Total epochs: {epochs}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write("\nTraining History:\n")
        for i, epoch in enumerate(history['epochs']):
            f.write(f"Epoch {epoch}: Train Loss {history['train_loss'][i]:.4f}, "
                   f"Val Loss {history['val_loss'][i]:.4f}, "
                   f"Val Acc {history['val_accuracy'][i]:.2f}%\n")
    
    train_logger.info(f"Training history saved to: {history_path}")
    train_logger.info("=" * 60)
    
    return history


def result_validation(
    pth_path: str, 
    model: nn.Module, 
    eval_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    model_name: str = "Model",
    num_classes: int = 8,
    save_confusion_matrix: bool = True,
    confusion_matrix_dir: str = "results/confusion_matrix_plots"
) -> dict:
    """
    Validate the model's performance on the evaluation dataset with comprehensive metrics.
    
    Parameters
    ----------
    pth_path : str
        Path to the saved model checkpoint.
    model : nn.Module
        The model to validate.
    eval_loader : DataLoader
        DataLoader for the evaluation dataset.
    criterion : nn.Module
        Loss function used for validation.
    device : torch.device
        Device to run the validation on (CPU or GPU).
    model_name : str, optional
        Name of the model for logging and saving plots.
    num_classes : int, optional
        Number of classes in the dataset.
    save_confusion_matrix : bool, optional
        Whether to save confusion matrix plot.
    confusion_matrix_dir : str, optional
        Directory to save confusion matrix plots.
    
    Returns
    -------
    dict
        Dictionary containing validation results including accuracy, F1 scores, etc.
    """
    # Create validation logger
    val_logger = get_logger(
        filename=f"logs/{model_name}_validation.log",
        name=f"{model_name}_VAL",
        level="INFO",
        overwrite=True
    )
    
    val_logger.info("=" * 50)
    val_logger.info("STARTING MODEL VALIDATION")
    val_logger.info("=" * 50)
    
    # Load model checkpoint
    val_logger.info(f"Loading model from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if 'accuracy' in checkpoint:
        val_logger.info(f"Model training accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Calculate model parameters and size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_file_size = os.path.getsize(pth_path) / (1024 * 1024)  # Convert to MB
    
    # Get system information
    system_info = {
        'CPU': psutil.cpu_count(),
        'Memory': psutil.virtual_memory().total / (1024**3),  # Convert to GB
        'Available Memory': psutil.virtual_memory().available / (1024**3),  # Convert to GB
    }
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        system_info['GPU'] = torch.cuda.get_device_name(0)
        system_info['GPU Memory'] = gpu_memory
    
    val_logger.info("MODEL AND SYSTEM INFORMATION")
    val_logger.info(f"Total Parameters: {total_params:,}")
    val_logger.info(f"Trainable Parameters: {trainable_params:,}")
    val_logger.info(f"Model File Size: {model_file_size:.2f} MB")
    val_logger.info(f"CPU Cores: {system_info['CPU']}")
    val_logger.info(f"Total Memory: {system_info['Memory']:.2f} GB")
    val_logger.info(f"Available Memory: {system_info['Available Memory']:.2f} GB")
    if 'GPU' in system_info:
        val_logger.info(f"GPU: {system_info['GPU']}")
        val_logger.info(f"GPU Memory: {system_info['GPU Memory']:.2f} GB")
    
    # Basic evaluation using ann_evaluate
    avg_loss = ann_evaluate(model, eval_loader, criterion, device)
    
    # Detailed evaluation with accuracy, F1 score and inference time
    model.eval()
    correct = 0
    total = 0
    all_preds, all_true = [], []
    inference_times = []
    
    val_logger.info("Performing detailed evaluation...")
    
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs = inputs.float().to(device)
            targets = targets.long().to(device)
            
            # Measure inference time
            start_time = time.time()
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                # Model returns tuple (outputs, _)
                outputs, _ = model(inputs)
            else:
                # Model returns only outputs
                outputs = model(inputs)
            end_time = time.time()
            
            batch_inference_time = (end_time - start_time) / inputs.size(0)  # Per sample
            inference_times.extend([batch_inference_time] * inputs.size(0))
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            preds = predicted.cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    f1_macro = f1_score(all_true, all_preds, average='macro')
    f1_weighted = f1_score(all_true, all_preds, average='weighted')
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to milliseconds
    
    val_logger.info(f"Validation Loss: {avg_loss:.4f}")
    val_logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    val_logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    val_logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    val_logger.info(f"Average Inference Time: {avg_inference_time:.4f} ms per sample")
    
    # Class-wise accuracy
    cm = confusion_matrix(all_true, all_preds, labels=range(num_classes))
    class_names = [str(i) for i in range(num_classes)]
    
    val_logger.info("Class-wise accuracy:")
    for i, cls in enumerate(class_names):
        cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        val_logger.info(f"Class {cls}: {cls_acc:.4f}")
    
    # Save confusion matrix
    if save_confusion_matrix:
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        
        ax = plot_confusion_matrix(
            test_y=all_true,
            pred_y=all_preds,
            class_names=class_names,
            normalize=True,
            fontsize=18
        )
        
        plt.title(f'Confusion Matrix – {model_name}\nAcc {accuracy:.2f}% | F1 {f1_macro:.4f}')
        fig_path = os.path.join(confusion_matrix_dir, f"{model_name}_validation_cm.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        val_logger.info(f"Confusion matrix saved to {fig_path}")
    
    # Summary
    val_logger.info("=" * 50)
    val_logger.info("VALIDATION SUMMARY")
    val_logger.info("=" * 50)
    val_logger.info(f"Model: {model_name}")
    val_logger.info(f"Total Parameters: {total_params:,}")
    val_logger.info(f"Model File Size: {model_file_size:.2f} MB")
    val_logger.info(f"Validation Loss: {avg_loss:.4f}")
    val_logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    val_logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    val_logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    val_logger.info(f"Average Inference Time: {avg_inference_time:.4f} ms per sample")
    val_logger.info("=" * 50)
    
    # Return results dictionary
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'inference_time_ms': avg_inference_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_file_size,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_true,
        'system_info': system_info
    }
    
    return results