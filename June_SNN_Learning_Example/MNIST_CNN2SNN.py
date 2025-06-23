import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron
from matplotlib import pyplot as plt


# -------------------------------- Dataset Preparation --------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]
)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



# -------------------------------- Model Definition --------------------------------
class CNN_MNIST(nn.Module):
    """
    Difine a CNN based model to classify MNIST digits.
    """
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pooling1(X)
        X = self.conv2(X)
        X = F.relu(X)
        X = self.pooling2(X)
        X = X.view(X.size(0), -1)  # Flatten the tensor
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)

        return X
    


# -------------------------------- Define Loss and Optimizer --------------------------------
model = CNN_MNIST()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



# -------------------------------- Training the Model --------------------------------
def train(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')
                running_loss = 0.0
        print(f'Epoch [{epoch+1}/{epochs}] completed.')

# train(model, trainloader, criterion, optimizer, epochs=10)
# print("Training completed.")



# -------------------------------- Evaluate the Model --------------------------------
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# evaluate(model, testloader)
# print("Evaluation completed.")



# -------------------------------- Save Current Model Weight --------------------------------
# model_save_path = r'model/mnist_cnn.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f"Model weights saved to {model_save_path}")




# -------------------------------- Transform Model to SNN --------------------------------

# Difine T steps
T = 10
PATH_CNN_MODEL = r'Efficient_ML_System/model/mnist_cnn.pth'

class SNN_MNIST_SpkingJelly(nn.Module):
    def __init__(self, cnn_path=None):
        super(SNN_MNIST_SpkingJelly, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.if1 = neuron.IFNode(v_threshold=1.0, v_reset=0.0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.if2 = neuron.IFNode(v_threshold=1.0, v_reset=0.0)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.if3 = neuron.IFNode(v_threshold=1.0, v_reset=0.0)
        self.fc2 = nn.Linear(128, 10)

        if cnn_path:
            cnn_state_dict = torch.load(cnn_path)
            self.load_state_weight(cnn_state_dict)

    def load_state_weight(self, cnn_state_dict):
        self.conv1.load_state_dict({k.replace('conv1.', ''): v for k, v in cnn_state_dict.items() if 'conv1' in k})
        self.conv2.load_state_dict({k.replace('conv2.', ''): v for k, v in cnn_state_dict.items() if 'conv2' in k})
        self.fc1.load_state_dict({k.replace('fc1.', ''): v for k, v in cnn_state_dict.items() if 'fc1' in k})
        self.fc2.load_state_dict({k.replace('fc2.', ''): v for k, v in cnn_state_dict.items() if 'fc2' in k})
        print("Model weights loaded successfully.")

    def forward(self, X):
        # X shape: (N, C, H, W)
        n = X.shape[0]
        # out_spike_counter = torch.zeros(n, 10, device=X.device)  # Counter spikes for fc2

        # reset membrane potential for all neurons
        self.if1.reset()
        self.if2.reset()
        self.if3.reset()

        for t in range(T):
            x_conv1 = self.conv1(X)
            x_pooling1 = self.pooling1(x_conv1)
            x_if1 = self.if1(x_pooling1)

            x_conv2 = self.conv2(x_if1)
            x_pooling2 = self.pooling2(x_conv2)
            x_if2 = self.if2(x_pooling2)

            x_flatten = x_if2.view(n, -1)  # Flatten the tensor
            x_fc1 = self.fc1(x_flatten)
            x_if3 = self.if3(x_fc1)

            if t == 0:
                out_potential_fc2 = self.fc2(x_if3)
            else:
                out_potential_fc2 += self.fc2(x_if3)
        
        return out_potential_fc2
    

def evaluate_snn(model, test_loader, num_steps, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    epoch_running_time = []
    total_start_time = time.time() # All epoch running time

    print('Evaluating SNN model...')
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            start_time = time.time()

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_end_time = time.time()
            epoch_running_time.append(batch_end_time - start_time)

            if (i + 1) % 20 == 0: 
                print(f'Test Batch [{i+1}/{len(test_loader)}]')

    total_runtime = time.time() - total_start_time
    print(f'Total evaluation time: {total_runtime:.2f} seconds')
    
    accuracy = 100 * correct / total
    print(f'Accuracy of the SNN on the test set: {accuracy:.2f}%')

    return accuracy, epoch_running_time

def train_snn(model, train_loader, criterion, optimizer, device='cpu', epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:    
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0 
        print(f'Epoch [{epoch+1}/{epochs}] completed.')

snn_model = SNN_MNIST_SpkingJelly(cnn_path=PATH_CNN_MODEL)





# -------------------------------- Test Functions --------------------------------
def test1(model, train_loader, test_loader, device='cpu', epochs=5):
    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Test1: Training SNN with pre-trained CNN weights(not training SNN model)')
    # train_snn(model, train_loader, criterion, optimizer, device=device, epochs=epochs) # Optical for we've pre-trained CNN model
    acc, epoch_run_time = evaluate_snn(model, test_loader, T, device=device)
    print(f'Accuracy after training: {acc:.2f}%')
    print(f'Epoch running time: {epoch_run_time}')
    print(f'Total evaluation time: {sum(epoch_run_time):.2f} seconds')

    # save acc and epoch_run_time to file
    with open('snn_mnist_results.txt', 'w') as f:
        f.write(f'Accuracy: {acc:.2f}%\n')
        f.write(f'Epoch running time: {epoch_run_time}\n')
    print('Results saved to snn_mnist_results.txt')

# test1(snn_model, trainloader, testloader, device='cpu', epochs=5) # test1

def test2(model, train_loader, test_loader, device='cpu', epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Test2: Training SNN with pre-trained CNN weights and training SNN model')
    train_snn(model, train_loader, criterion, optimizer, device=device, epochs=epochs)
    acc, epoch_run_time = evaluate_snn(model, test_loader, T, device=device)
    print(f'Accuracy after training: {acc:.2f}%')
    print(f'Epoch running time: {epoch_run_time}')
    print(f'Total evaluation time: {sum(epoch_run_time):.2f} seconds')

    # save acc and epoch_run_time to file
    with open('snn_mnist_results.txt', 'a') as f:
        f.write(f'Accuracy after training: {acc:.2f}%\n')
        f.write(f'Epoch running time: {epoch_run_time}\n')
    print('Results saved to snn_mnist_results.txt')

test2(snn_model, trainloader, testloader, device='cpu', epochs=5) # test2