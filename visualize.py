import sys
import os
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
from models.ViT import ViT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

#获取第一张图片
test_set = dset.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
image_tensor, label = test_set[0] 
raw_img, _ = dset.CIFAR10(root='./datasets', train=False, download=True)[0]

#初始化模型并加载权重
model = ViT(
    img_size=32, patch_size=4, in_channels=3, 
    embed_dim=256, head_num=8, mlp_dim=512, num_layers=4, num_classes=10
).to(DEVICE)
model.load_state_dict(torch.load('vit_cifar10.pth', map_location=DEVICE))
model.eval()

img_input = image_tensor.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    _ = model(img_input)

#取出最后一层Transformer的注意力权重
attention_matrix = model.transformer_blocks[-1].attention.attn_weights
attention_matrix = attention_matrix.squeeze(0).cpu().numpy() 
cls_attention = attention_matrix[:, 0, 1:] 
grid_size = int(np.sqrt(cls_attention.shape[1]))
cls_attention = cls_attention.reshape(8, grid_size, grid_size)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

axes[0].imshow(raw_img)
axes[0].set_title(f'Original Image (Label: {label})')
axes[0].axis('off')

for i in range(8):
    ax = axes[i+1]
    img_plot = ax.imshow(cls_attention[i], cmap='viridis', interpolation='bicubic')
    ax.set_title(f'Head {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()