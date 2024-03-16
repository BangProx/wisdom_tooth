# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import f1_score

import os
import pandas as pd
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 전처리
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#데이터셋 로드
full_dataset = ImageFolder(root='path', transform=transform)

train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size

train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=2)

model = models.resnet50(pretrained=True)  # Using a simpler model, ResNet18
num_ftrs = model.fc.in_features
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

model.fc = nn.Sequential(
    nn.Flatten(),  # Flatten the output of the GAP layer
    nn.Linear(num_ftrs, 128),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(128, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}")

    # Validation
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"Validation Loss: {validation_loss/len(validation_loader)}, Accuracy: {100 * correct / total}%, F1-Score: {epoch_f1:.4f}")

print('Finished Training and Validation')