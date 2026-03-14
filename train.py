import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ViT import ViT
import torchvision.datasets as dset
import torchvision.transforms as T

#定义超参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#准备数据集
transform = T.Compose([
    T.ToTensor(),  
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])
print("开始下载/加载训练集...")
cifar10_train = dset.CIFAR10(root='./datasets', train=True, download=True, transform=transform)

train_loader = DataLoader(cifar10_train, batch_size=64, shuffle=True) 
 

# 定义模型、优化器和损失函数
model = ViT(
    img_size=32,    
    patch_size=4,   
    in_channels=3, 
    embed_dim=256,   
    head_num=8,      
    mlp_dim=512,     
    num_layers=4,    
    num_classes=10   
).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

#训练模型
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

torch.save(model.state_dict(), 'vit_cifar10.pth')
