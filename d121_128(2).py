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
        self.model = torchvision.models.densenet121(pretrained=True)
        n_features = self.model.classifier.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
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
    transforms.Resize((512, 512)),           # 이미지 크기를 512x512로 조정
    transforms.RandomHorizontalFlip(p=0.5),  # 이미지를 50% 확률로 뒤집기
    transforms.RandomResizedCrop(512, scale=(0.9, 1.0)),  # 스케일 범위 0.9에서 1.0으로 무작위 크롭
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기와 대비를 0.8에서 1.2 범위 내에서 무작위로 조정
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''
transforms.RandomResizedCrop(224),
    transforms.RandomRotation(50),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
'''
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data
df = pd.read_csv(f'path')

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
best_model_path = 'path'
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

#121_512_lr 결과
'''
Epoch 00026: reducing learning rate of group 0 to 2.5000e-05.
Epoch 26/50, Train Loss: 0.4130, Validation Loss: 0.5620, Validation Metric: 0.7502
100%|██████████| 65/65 [00:34<00:00,  1.90it/s]
100%|██████████| 17/17 [00:07<00:00,  2.33it/s]
Epoch 27/50, Train Loss: 0.3852, Validation Loss: 0.5586, Validation Metric: 0.7475
100%|██████████| 65/65 [00:34<00:00,  1.90it/s]
100%|██████████| 17/17 [00:06<00:00,  2.60it/s]
Epoch 28/50, Train Loss: 0.3844, Validation Loss: 0.5689, Validation Metric: 0.7680
'''

#121_256 leakyrelu결과
'''
00%|██████████| 17/17 [00:06<00:00,  2.65it/s]
Epoch 24/50, Train Loss: 0.4452, Validation Loss: 0.5853, Validation Metric: 0.7332
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:07<00:00,  2.19it/s]
Epoch 00025: reducing learning rate of group 0 to 2.5000e-05.
Epoch 25/50, Train Loss: 0.4409, Validation Loss: 0.5959, Validation Metric: 0.6975
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:06<00:00,  2.74it/s]
Epoch 26/50, Train Loss: 0.4205, Validation Loss: 0.5489, Validation Metric: 0.7365
100%|██████████| 65/65 [00:33<00:00,  1.96it/s]
100%|██████████| 17/17 [00:07<00:00,  2.23it/s]
Epoch 27/50, Train Loss: 0.4009, Validation Loss: 0.5577, Validation Metric: 0.7596
100%|██████████| 65/65 [00:33<00:00,  1.94it/s]
100%|██████████| 17/17 [00:06<00:00,  2.63it/s]Early stopping triggered at epoch 28

'''

#121_256 do=0.2 elu
'''
Epoch 20/50, Train Loss: 0.4569, Validation Loss: 0.5304, Validation Metric: 0.7235
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:07<00:00,  2.27it/s]
Epoch 21/50, Train Loss: 0.4394, Validation Loss: 0.6129, Validation Metric: 0.7036
100%|██████████| 65/65 [00:34<00:00,  1.89it/s]
100%|██████████| 17/17 [00:06<00:00,  2.72it/s]
Epoch 22/50, Train Loss: 0.4443, Validation Loss: 0.5525, Validation Metric: 0.7231
100%|██████████| 65/65 [00:33<00:00,  1.95it/s]
100%|██████████| 17/17 [00:07<00:00,  2.16it/s]
Epoch 23/50, Train Loss: 0.4262, Validation Loss: 0.5476, Validation Metric: 0.7446
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:06<00:00,  2.72it/s]
Epoch 00024: reducing learning rate of group 0 to 2.5000e-05.
Epoch 24/50, Train Loss: 0.4275, Validation Loss: 0.5629, Validation Metric: 0.7289
100%|██████████| 65/65 [00:33<00:00,  1.95it/s]
100%|██████████| 17/17 [00:07<00:00,  2.19it/s]
Epoch 25/50, Train Loss: 0.4243, Validation Loss: 0.5510, Validation Metric: 0.7208
100%|██████████| 65/65 [00:33<00:00,  1.92it/s]
100%|██████████| 17/17 [00:06<00:00,  2.62it/s]
Epoch 26/50, Train Loss: 0.3944, Validation Loss: 0.5992, Validation Metric: 0.7063
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:07<00:00,  2.25it/s]Early stopping triggered at epoch 27
'''

#121_256 do = 0.5 elu
'''
Epoch 34/50, Train Loss: 0.3525, Validation Loss: 0.6333, Validation Metric: 0.7375
100%|██████████| 65/65 [00:32<00:00,  1.97it/s]
100%|██████████| 17/17 [00:06<00:00,  2.56it/s]
Epoch 35/50, Train Loss: 0.3638, Validation Loss: 0.6692, Validation Metric: 0.7260
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:07<00:00,  2.35it/s]
Epoch 36/50, Train Loss: 0.3548, Validation Loss: 0.6519, Validation Metric: 0.7460
100%|██████████| 65/65 [00:33<00:00,  1.91it/s]
100%|██████████| 17/17 [00:06<00:00,  2.67it/s]
Epoch 37/50, Train Loss: 0.3735, Validation Loss: 0.6584, Validation Metric: 0.7189
100%|██████████| 65/65 [00:33<00:00,  1.92it/s]
100%|██████████| 17/17 [00:07<00:00,  2.24it/s]
Epoch 00038: reducing learning rate of group 0 to 2.5000e-05.
Epoch 38/50, Train Loss: 0.3348, Validation Loss: 0.6902, Validation Metric: 0.7232
100%|██████████| 65/65 [00:33<00:00,  1.91it/s]
100%|██████████| 17/17 [00:06<00:00,  2.68it/s]
Epoch 39/50, Train Loss: 0.3275, Validation Loss: 0.6749, Validation Metric: 0.7249
100%|██████████| 65/65 [00:33<00:00,  1.92it/s]
100%|██████████| 17/17 [00:07<00:00,  2.18it/s]
Epoch 40/50, Train Loss: 0.3174, Validation Loss: 0.6779, Validation Metric: 0.7272
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:06<00:00,  2.66it/s]Early stopping triggered at epoch 41
'''

#121_128 do = 0.5 elu
'''
100%|██████████| 65/65 [00:33<00:00,  1.93it/s]
100%|██████████| 17/17 [00:06<00:00,  2.80it/s]
Epoch 20/50, Train Loss: 0.4628, Validation Loss: 0.5346, Validation Metric: 0.7535
100%|██████████| 65/65 [00:33<00:00,  1.94it/s]
100%|██████████| 17/17 [00:07<00:00,  2.20it/s]
Epoch 21/50, Train Loss: 0.4419, Validation Loss: 0.5652, Validation Metric: 0.7362
100%|██████████| 65/65 [00:33<00:00,  1.94it/s]
100%|██████████| 17/17 [00:06<00:00,  2.82it/s]
Epoch 22/50, Train Loss: 0.4578, Validation Loss: 0.5540, Validation Metric: 0.7559
100%|██████████| 65/65 [00:33<00:00,  1.95it/s]
100%|██████████| 17/17 [00:07<00:00,  2.41it/s]
Epoch 23/50, Train Loss: 0.4223, Validation Loss: 0.5690, Validation Metric: 0.7439
100%|██████████| 65/65 [00:32<00:00,  1.99it/s]
100%|██████████| 17/17 [00:06<00:00,  2.56it/s]
Epoch 24/50, Train Loss: 0.4354, Validation Loss: 0.5588, Validation Metric: 0.7548
100%|██████████| 65/65 [00:32<00:00,  1.99it/s]
100%|██████████| 17/17 [00:05<00:00,  2.85it/s]
Epoch 00025: reducing learning rate of group 0 to 2.5000e-05.
Epoch 25/50, Train Loss: 0.4225, Validation Loss: 0.6056, Validation Metric: 0.7607
100%|██████████| 65/65 [00:33<00:00,  1.97it/s]
100%|██████████| 17/17 [00:07<00:00,  2.23it/s]
Epoch 26/50, Train Loss: 0.4023, Validation Loss: 0.6140, Validation Metric: 0.7315
100%|██████████| 65/65 [00:33<00:00,  1.96it/s]
100%|██████████| 17/17 [00:06<00:00,  2.80it/s]
Epoch 27/50, Train Loss: 0.3945, Validation Loss: 0.5884, Validation Metric: 0.7552
100%|██████████| 65/65 [00:33<00:00,  1.96it/s]
100%|██████████| 17/17 [00:06<00:00,  2.44it/s]Early stopping triggered at epoch 28
'''

#d121_128 do = 0.2
"""
Epoch 11/50, Train Loss: 0.5444, Validation Loss: 0.5227, Validation Metric: 0.7427
100%|██████████| 65/65 [00:33<00:00,  1.96it/s]
100%|██████████| 17/17 [00:08<00:00,  1.94it/s]
Epoch 12/50, Train Loss: 0.5582, Validation Loss: 0.5946, Validation Metric: 0.6993
100%|██████████| 65/65 [00:32<00:00,  1.99it/s]
100%|██████████| 17/17 [00:06<00:00,  2.56it/s]
Epoch 13/50, Train Loss: 0.5224, Validation Loss: 0.5409, Validation Metric: 0.7447
100%|██████████| 65/65 [00:39<00:00,  1.64it/s]
100%|██████████| 17/17 [00:07<00:00,  2.26it/s]
Epoch 14/50, Train Loss: 0.5191, Validation Loss: 0.5095, Validation Metric: 0.7343
100%|██████████| 65/65 [00:32<00:00,  1.97it/s]
100%|██████████| 17/17 [00:06<00:00,  2.57it/s]
Epoch 15/50, Train Loss: 0.5123, Validation Loss: 0.5292, Validation Metric: 0.7109
100%|██████████| 65/65 [00:32<00:00,  1.98it/s]
100%|██████████| 17/17 [00:08<00:00,  2.12it/s]
Epoch 16/50, Train Loss: 0.5237, Validation Loss: 0.5215, Validation Metric: 0.7125
100%|██████████| 65/65 [00:33<00:00,  1.97it/s]
100%|██████████| 17/17 [00:06<00:00,  2.56it/s]
"""