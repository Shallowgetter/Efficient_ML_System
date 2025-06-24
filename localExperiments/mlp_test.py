import sys
import os
import datetime
import yaml
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.mlp import MLP
from preprocessing.getDataset import get_npz_dataloader, get_dataloader
from utils.utils import get_logger, save_checkpoint, AverageMeter

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="MLP Training Script")
    parser.add_argument('--config', type=str, default='localExperiments/model_param/mlp_cnn_params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_name', type=str, default='MLP_test_1',
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
    """
    # Setup directories
    logs_dir = "localExperiments/logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    checkpoints_dir = "localExperiments/model_result/checkpoints"
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
    elif config['dataLoader'] == 'get_dataloader':
        data_loader_func = get_dataloader
    else:
        raise ValueError(f"Unknown dataLoader: {config['dataLoader']}")
    
    train_loader = data_loader_func(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_train.npz",
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    test_loader = data_loader_func(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_test.npz",
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

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
        val_loss, val_accuracy = evaluate(model, test_loader, device, criterion)
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

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config, args.model_name)
    train_model(config)


