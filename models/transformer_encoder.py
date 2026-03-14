import torch
from torch import nn
class Attention(nn.Module):
    def __init__(self, embed_dim, head_num):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.qkv_projection = nn.Linear(embed_dim, 3 * embed_dim) #线性变换得到q、k、v
        self.score_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, __ = x.size() #(batch_size, seq_len, embed_dim) (n,197,768)
        qkv = self.qkv_projection(x)
        head_dim = self.embed_dim // self.head_num
        qkv = qkv.view(batch_size, seq_len, self.head_num, 3 , head_dim)
        qkv = qkv.permute(3, 0, 2, 1, 4) #调整维度顺序为(3, batch_size, head_num, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] #得到(batch_size, head_num, seq_len, head_dim)的q、k、v   (n,12,197,64)
        weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = torch.softmax(weights, dim=-1)
        self.attn_weights = weights.detach() #保存权重，便于可视化注意力机制
        scores = torch.matmul(weights, v)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim) #保持输出与输入的形状相同
        scores = self.score_projection(scores) #得到最终的输出
        return scores

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_num, mlp_dim):
        super().__init__()
        self.attention = Attention(embed_dim, head_num)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x
