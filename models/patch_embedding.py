import torch
from torch import nn
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        x = self.projection(x)
        x = torch.flatten(x, start_dim=2) #如果输入(n,3,224,224)，输出(n,embed_dim,196)
        x = x.transpose_(1, 2) #(n,196,embed_dim)
        return x
