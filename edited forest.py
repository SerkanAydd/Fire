import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define your CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes = 3):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)  # Adding Batch Normalization
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)  # Adding Batch Normalization
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)  # Adding Batch Normalization
        self.dropout4 = nn.Dropout2d(p=0.1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 3)
        

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool1(x)
        
        x = F.leaky_relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = F.leaky_relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x = self.pool3(x)
        
        x = F.leaky_relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.pool4(x)
        
        x = x.view(-1, 128 * 8 * 8)  # Reshape for fully connected layer
        x = F.leaky_relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)  # Apply softmax for output layer
        return x
    

# Data paths
train_dataset_path = "/kaggle/input/forest-fire-smoke-and-non-fire-image-dataset/FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET/train"
test_dataset_path = "/kaggle/input/forest-fire-smoke-and-non-fire-image-dataset/FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET/test"

# Transformations
mean = [0.4183, 0.3783, 0.3330]
std = [0.2347, 0.2152, 0.2055]

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to set device
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

# Training function
def train_nn(model, train_loader, test_loader, criterion, optimizer, scheduler, n_epochs):
    device = set_device()
    print(device)
    model.to(device)
    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1))
        model.train()
        running_loss = 0
        running_correct = 0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * running_correct / total
        print("    - Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))
        
        scheduler.step(evaluate_model_on_test_set(model, test_loader, criterion))

    print("Finished")
    return model

# Test function
def evaluate_model_on_test_set(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    epoch_acc = 100 * predicted_correctly_on_epoch / total
    print("    - Testing dataset. Got %d out of %d images correctly (%.3f%%)"
          % (predicted_correctly_on_epoch, total, epoch_acc)
          )
    
#    accuracy = 100 * predicted_correctly_on_epoch / total
    test_loss /= len(test_loader)
    
    return test_loss



# Initialize the model, loss function, and optimizer
custom_cnn_model = CustomCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_cnn_model.parameters(), lr=0.001, weight_decay=0.0025)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

train_nn(custom_cnn_model, train_loader, test_loader, loss_fn, optimizer, scheduler, 50)
