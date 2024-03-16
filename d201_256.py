import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score


class BaselineDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.df['risk'] = self.df['risk'].apply(lambda x: 1 if x == 'high' else 0)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    #여기서 전처리 시도
    def __getitem__(self, idx):
        img_name, label = self.df.iloc[idx]
        img_fname = f'path/{img_name}'
        img = Image.open(img_fname)

        if self.transform:
            img = self.transform(img)

        return img, label


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        #여기서 모델명 바꾸자
        #sequential도 수정할 수 있으면 해보자
        self.model = torchvision.models.densenet201(pretrained=True)
        n_features = self.model.classifier.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.model.classifier = self.fc

    def forward(self, x):
        x = self.model(x)
        return x


# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer.zero_grad()

        outputs = model(inputs).view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


# Define the validation function
def valid(model, val_loader, criterion, device):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs).view(-1)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = torch.sigmoid(outputs).round()
            metrics.append(f1_score(labels.cpu(), preds.cpu(), average='macro'))
    return losses, metrics


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 50
num_batches = 32

# transformations
#train꺼는 더 Augmentation적용해보자
train_transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomRotation(50),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data
df = pd.read_csv(f'/content/drive/MyDrive/AI/asn/train.csv')

# train / validation split with stratified sampling
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20231101)

for train_idx, val_idx in skf.split(df, df['risk']):
    # use first fold as validation set
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    break

# prepare dataset
train_dataset = BaselineDataset(train_df, transform=train_transform)
val_dataset = BaselineDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=num_batches, shuffle=False)

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# 모델, EarlyStopping 인스턴스, optimizer 및 scheduler 초기화
# criterion crossentrophy적용해보고 lr도 수정해보자
model = BaselineModel().to(device)
early_stopping = EarlyStopping(patience=15, min_delta=0.001)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float('inf')
best_model_path = '/content/drive/MyDrive/AI/tooth_weights/dense121.pth'
# 학습 루프
for epoch in range(num_epochs):
    train_losses = train(model, train_loader, criterion, optimizer, device)
    val_losses, val_metrics = valid(model, val_loader, criterion, device)
    val_loss_avg = np.mean(val_losses)

    # ReduceLROnPlateau 업데이트
    scheduler.step(val_loss_avg)

    # EarlyStopping 체크
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), best_model_path)
    early_stopping(val_loss_avg)
    if early_stopping.early_stop:
        print("Early stopping triggered at epoch {}".format(epoch+1))
        break

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, Validation Loss: {val_loss_avg:.4f}, Validation Metric: {np.mean(val_metrics):.4f}")