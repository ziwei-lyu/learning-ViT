import torch
import torch.nn as nn
import torch.nn.functional as F
from .patch_embedding import PatchEmbedding
from .transformer_encoder import TransformerBlock
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, head_num=12, mlp_dim=3072, num_layers=12, num_classes=1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, head_num, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) #添加一个cls_token作为分类标记
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim)) #位置编码

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) #将cls_token扩展到(batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) #将cls_token与patch_embedding拼接
        x = x + self.pos_embedding #添加位置编码
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0] #取出cls_token对应的输出
        out = self.classifier(cls_token_final) #通过分类器得到最终输出
        return out

