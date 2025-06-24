import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.mlp import MLP
from preprocessing.getDataset import get_npz_dataloader
from utils.utils import get_logger, save_checkpoint, AverageMeter


train_loader = get_npz_dataloader(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_train.npz",
        batch_size=1024,
        shuffle=True,
        num_workers=0
    )


test_loader = get_npz_dataloader(
        npz_path="/Users/xiangyifei/Documents/GitHub/efficientComputingSystem/data/shl_test.npz",
        batch_size=1024,
        shuffle=False,
        num_workers=0
    )



model = MLP(input_size=10800)
print(model)
model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)


# logs directory
logs_dir = "localExperiments/logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


# logger initialization, using logs directory
logger = get_logger(
    filename=os.path.join(logs_dir, "mlp_test.log"),
    name="MLPTestLogger",
    level="DEBUG",
    overwrite=True,
    to_stdout=True
)



def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_meter = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    
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



checkpoints_dir = "localExperiments/model_result/checkpoints"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
    logger.info(f"Created checkpoints directory for MLP experiments: {checkpoints_dir}")



logger.info(f"Train dataset has {len(train_loader)} batches with batch size {train_loader.batch_size}")

best_accuracy = 0.0
for epoch in range(100):
    logger.info(f"Epoch {epoch+1}/100")

    train_loss_meter = AverageMeter()
    batch_count = 0
    
    for i, (inputs, targets) in enumerate(train_loader):
        batch_count += 1
        inputs = inputs.float().to(device)
        targets = targets.long().to(device)

        # check the shape of input and target
        # print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = criterion(outputs, targets)
        loss_value.backward()
        optimizer.step()
        
        train_loss_meter.update(loss_value.item(), inputs.size(0))
        
        if i % 100 == 0:
            logger.info(f"Epoch {epoch+1}, Step {i}, Batch count: {batch_count}, Loss: {loss_value.item():.4f}")
    
    logger.info(f"Epoch {epoch+1} had {batch_count} batches")
    
    
    val_loss, val_accuracy = evaluate(model, test_loader, device)
    logger.info(f"Epoch {epoch+1} completed. Train Loss: {train_loss_meter.avg:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    
    is_best = val_accuracy > best_accuracy
    best_accuracy = max(val_accuracy, best_accuracy)
    
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch+1,
        loss=train_loss_meter.avg,
        accuracy=val_accuracy,
        filename=os.path.join(checkpoints_dir, f"mlp_checkpoint_epoch{epoch+1}.pth"),
        is_best=is_best,
        best_filename=os.path.join(checkpoints_dir, "mlp_best.pth")
    )

logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")

# summary log file
summary_path = os.path.join(logs_dir, "mlp_train_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Train completed at: {datetime.datetime.now()}\n")
    f.write(f"Best validation accuracy: {best_accuracy:.2f}%\n")
    f.write(f"Model saved at: {os.path.join(checkpoints_dir, 'mlp_best.pth')}\n")


