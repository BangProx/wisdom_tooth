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

from google.colab import drive
drive.mount('/content/drive')

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

model = models.inception_v3(pretrained=True)  

#efficientnetv2dpsms 이미 GAP 포함됨
num_ftrs = model.last_linear.in_features
intermediate_features =128

model.last_linear = nn.Sequential(
    nn.Flatten(),  
    nn.Linear(num_ftrs, 128),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)

criterion = nn.CrossEntropyLoss()
# 컴파일
optimizer = optim.Adam(model.parameters(), lr=0.0001)


#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

"""1. batch_size 64 => 32
  현재 배치크기만 조절하는 걸로는 학습을 반복할수록 성능이 줄어든다
  
2. 과적합인 거 같아서 drop out + linear 적용
3. 그래도 안 나아져서 학습률 0.001=>0.0001로 적용
4. ReduceLROnPlateau 적용
5. adam 적용 결과 오히려 결과 감소,
일단 다시 batch_size 다시 조절 해보자
1) 16, 2) 64
"""

# Training
num_epochs = 10
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