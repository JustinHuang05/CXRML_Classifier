from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna

# Define the model class
class MLModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MLModel, self).__init__()
        # Use a pre-trained DenseNet121 model from timm
        self.base_model = timm.create_model('densenet121', pretrained=True)
        # Change the first convolutional layer to accept single-channel input
        self.base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Replace the final fully connected layer with a new one that includes Dropout for regularization
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model.num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader.dataset))
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loader_tqdm = tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}', unit='batch')
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loader_tqdm.set_postfix(loss=val_loss / len(val_loader.dataset))
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved to best_model.pth')
        else:
            epochs_without_improvement += 1
            # Early stopping condition
            if epochs_without_improvement >= patience:
                print('Early stopping triggered')
                break

        # Step the scheduler
        scheduler.step()  # For CosineAnnealingLR, step after each epoch

    # Plot and save the loss over epochs
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    output_file = 'loss_over_epochs.png'
    plt.savefig(output_file)
    plt.show()

# Hyperparameter tuning with Optuna
def objective(trial):
    # Define hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    
    # Initialize model, loss function, and optimizer
    model = MLModel(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler with CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Combine training and validation datasets for cross-validation
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    # Extract targets for stratified K-Fold
    targets = []
    for dataset in combined_dataset.datasets:
        targets.extend(dataset.targets)
    targets = np.array(targets)

    # Perform k-fold cross-validation
    k_folds = 5
    results = []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f'Fold {fold+1}/{k_folds}')
        
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(combined_dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(combined_dataset, batch_size=32, sampler=val_subsampler)
        
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, patience=5)
        
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        results.append(val_loss)
    
    avg_val_loss = np.mean(results)
    return avg_val_loss

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    return val_loss, val_accuracy

if __name__ == "__main__":
    # Define transforms with augmentation for training data, including conversion to grayscale
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transforms for validation data, including conversion to grayscale
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Path to the dataset
    dataPath = '/Users/justinhuang/Documents/Developer/ML/CXRML/CXRData'
    base_dir = Path(dataPath)
    train_dir = base_dir / 'train'
    valid_dir = base_dir / 'valid'

    # Load datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=valid_dir, transform=val_transform)
    
    # Combine datasets for cross-validation
    combined_dataset = ConcatDataset([train_dataset, val_dataset])

    # Define the study for hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f'Best hyperparameters: {study.best_params}')
    
    # Train the model with the best hyperparameters on the combined dataset
    best_model = MLModel(num_classes=3)
    best_optimizer = optim.Adam(best_model.parameters(), lr=study.best_params['lr'], weight_decay=study.best_params['weight_decay'])
    best_scheduler = optim.lr_scheduler.CosineAnnealingLR(best_optimizer, T_max=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define data loaders with the entire combined dataset for final training
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    
    # Train the model on the entire combined dataset
    train_model(best_model, combined_loader, DataLoader(val_dataset, batch_size=32, shuffle=False), criterion, best_optimizer, best_scheduler, device, num_epochs=25, patience=5)
