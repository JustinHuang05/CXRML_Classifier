from pathlib import Path
import random
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

class MLModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MLModel, self).__init__()
        self.base_model = timm.create_model('densenet121', pretrained=True)
        # Change the first convolutional layer to accept single-channel input
        self.base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Replace the final fully connected layer with a new one that includes Dropout for regularization
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def imshow(ax, inp, prediction, correctness, correct_label=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp, cmap='gray')
    
    ax.set_title(prediction, fontsize=12)
    if correctness:
        ax.set_xlabel('Correct!', color='green')
    else:
        ax.set_xlabel(f'Incorrect, Answer: {correct_label}', color='red')

    ax.set_xticks([])
    ax.set_yticks([])

def create_slideshow(predictions, class_names, test_accuracy, output_file):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    def show_image(i):
        ax.clear()
        if i < 5:
            ax.text(0.5, 0.5, f'Test Accuracy: {test_accuracy:.4f}', 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    fontsize=20, 
                    transform=ax.transAxes)
        else:
            img, label, correct, true_label = predictions[i - 5]
            prediction = f'Predicted: {class_names[label]}'
            correctness = correct
            correct_label = class_names[true_label]
            imshow(ax, img, prediction, correctness, correct_label if not correctness else None)
        fig.canvas.draw()

    def update(frame):
        show_image(frame)
        return ax

    ani = FuncAnimation(fig, update, frames=len(predictions) + 5, repeat=True, blit=False)
    ani.save(output_file, writer=PillowWriter(fps=0.5))  # 0.5 fps means 2 seconds per frame
    plt.close()

def test_model(model, test_loader, criterion, device, class_names, output_file):
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc='Testing', unit='batch')
        for inputs, labels in test_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader_tqdm.set_postfix(loss=test_loss / total, accuracy=correct / total)

            for i in range(inputs.size(0)):
                img = inputs.cpu().data[i]
                label = predicted.cpu().data[i]
                correct_prediction = (label == labels.cpu().data[i])
                true_label = labels.cpu().data[i]
                predictions.append((img, label, correct_prediction, true_label))

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    #create_slideshow(predictions, class_names, test_accuracy, output_file)

def main():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataPath = '/Users/justinhuang/Documents/Developer/ML/CXRML'
    base_dir = Path(dataPath)
    test_dir = base_dir / 'CXRData/test'
    output_file = base_dir / 'predictions.gif'

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    class_names = test_dataset.classes

    indices = random.sample(range(len(test_dataset)), 300)
    test_subset = Subset(test_dataset, indices)

    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    model = MLModel(num_classes=3)

    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    test_model(model, test_loader, criterion, device, class_names, output_file)

if __name__ == "__main__":
    main()
