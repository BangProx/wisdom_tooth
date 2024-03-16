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
train_data = ImageFolder(root='path', transform=transform)
val_data = ImageFolder(root='path', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

import torch
import torch.nn as nn
from torchvision import models

# 사전 훈련된 ResNet-18 모델 로드
model = models.resnet18(pretrained=True)

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet50, self).__init__()
        # 사전 훈련된 ResNet50 모델 로드
        self.resnet = models.resnet50(pretrained=True)

        # 최종 완전 연결 레이어 제거
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))

        # 전역 평균 풀링 추가
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 세 개의 완전 연결 계층 추가
        self.fc1 = nn.Linear(2048, 1024)  # 첫 번째 완전 연결 계층
        self.fc2 = nn.Linear(1024, 512)   # 두 번째 완전 연결 계층
        self.fc3 = nn.Linear(512, num_classes)  # 세 번째 완전 연결 계층, 분류를 위함

    def forward(self, x):
        # 입력을 ResNet 계층을 통해 전달
        x = self.resnet(x)

        # 전역 평균 풀링
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # 텐서 펴기

        # 완전 연결 계층을 통해 전달
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

num_ftrs = model.fc.in_features
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#model.fc = nn.Linear(num_ftrs, 2)
model.fc = nn.Sequential(
    nn.Flatten(),  
    nn.Linear(num_ftrs, 128),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Dropout(0.4),
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
        for inputs, labels in val_loader:
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
    print(f"Validation Loss: {validation_loss/len(val_loader)}, Accuracy: {100 * correct / total}%, F1-Score: {epoch_f1:.4f}")

print('Finished Training and Validation')