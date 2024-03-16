import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models
import pandas as pd
from torch.utils.data.dataset import random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

class HR_LR_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_filename).convert('RGB')
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))  # 엣지 강조

        if self.transform:
            image = self.transform(image)

        risk = self.data.iloc[idx, 1]
        label = int(risk == 'high')
        return image, label

# Specify the path to your CSV file
csv_file_path = "path"
test_csv_file = "path"

# Define transformations for the data
transform = transforms.Compose([
    transforms.Resize(224),
    # Resize images to fit ResNet-50 input size
    transforms.RandomHorizontalFlip(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset using HR_LR_Dataset
dataset = HR_LR_Dataset(root_dir='path', csv_file=csv_file_path, transform=transform)
print(len(dataset))
# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

num_classes = 2
# Load pre-trained DenseNet121
model_densenet121 = models.densenet121(pretrained=True)

# Modify the classifier for binary classification
num_ftrs = model_densenet121.classifier.in_features
model_densenet121.classifier = nn.Linear(num_ftrs, 2)

nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(64, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_densenet121.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_densenet121.parameters(), lr=0.001)#0.0001
early_stopping_counter = 0
early_stopping_patience = 20
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Training loop
num_epochs = 30  # Replace with the number of epochs you want to train for

best_loss = 100
for epoch in range(num_epochs):
    model_densenet121.train()  # Set the model to training mode
    running_loss = 0.0
    data_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model_densenet121(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate validation loss
    model_densenet121.eval()  # Set the model to evaluation mode
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_densenet121(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

    # Update the learning rate scheduler based on validation loss
    scheduler.step(validation_loss)

    # Check for early stopping
    if validation_loss < best_loss:
        best_loss = validation_loss
        early_stopping_counter = 0
        # Save the model weights to the "weights" directory
        model_path = 'path'
        torch.save(model_densenet121.state_dict(), model_path)
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
    # Calculate training accuracy
    model_densenet121.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_densenet121(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)  # Calculate F1 score

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Validation Accuracy: {accuracy * 100:.2f}%, Validation F1 Score: {f1 * 100:.2f}%")

print("Training complete")