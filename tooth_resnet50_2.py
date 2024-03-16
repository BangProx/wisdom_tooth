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

import torch
from torchvision import transforms

# 데이터 증강 파이프라인 정의
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 이미지를 50% 확률로 뒤집기
    transforms.Resize((224, 224)),           # 이미지 크기를 512x512로 조정
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 스케일 범위 0.9에서 1.0으로 무작위 크롭
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기와 대비를 0.8에서 1.2 범위 내에서 무작위로 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# 전처리
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#데이터셋 로드
train_data = ImageFolder(root='path', transform=train_transform)
val_data = ImageFolder(root='path', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

import torch
import torch.nn as nn
from torchvision import models

# 사전 훈련된 ResNet-50 모델 로드
resnet50 = models.resnet50(pretrained=True)

# ResNet-50 모델을 논문의 설명에 맞게 수정
# 최종 레이어를 전역 평균 풀링으로 대체한 후 세 개의 완전 연결 계층 추가
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
        self.fc1 = nn.Linear(2048, 512)  # 첫 번째 완전 연결 계층
        self.fc2 = nn.Linear(512, 256)  # 두 번째 완전 연결 계층
        self.fc3 = nn.Linear(256, num_classes)  # 세 번째 완전 연결 계층, 분류를 위함

    def forward(self, x):
        # 입력을 ResNet 계층을 통해 전달
        x = self.resnet(x)

        # 전역 평균 풀링
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # 텐서 펴기

        # 완전 연결 계층을 통해 전달
        #x = nn.functional.relu(self.fc0(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 사용 예시
num_classes = 2  # 클래스 수 정의 (논문에서의 PDS 값)
modified_resnet50 = ModifiedResNet50(num_classes=num_classes)

# 수정된 모델 출력
print(modified_resnet50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modified_resnet50.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modified_resnet50.to(device)

# Training
num_epochs = 30
for epoch in range(num_epochs):
    modified_resnet50.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = modified_resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}")

    # Validation
    modified_resnet50.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modified_resnet50(inputs)
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