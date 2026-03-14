import sys
import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.ViT import ViT

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = T.Compose([
    T.ToTensor(),  
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])

print("开始加载测试集...")
cifar10_test = dset.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
test_loader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, shuffle=False)   


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


try:
    model.load_state_dict(torch.load('vit_cifar10.pth'))
    print("成功加载模型权重 vit_cifar10.pth！")
except FileNotFoundError:
    print("找不到权重文件！请确认你已经跑完了 train.py 并生成了 .pth 文件。")
    sys.exit()

model.eval() 

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算最终准确率
accuracy = 100 * correct / total
print(f"模型在 10000 张测试集上的准确率: {accuracy:.2f}%")
