# learning-ViT

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)

## 📌 项目简介

这是我在了解深度学习的过程中，为了深入学习 [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) 架构而做的小练习。本项目提供了一个基于 PyTorch 构建的 Vision Transformer (ViT) 实现。代码不依赖第三方视觉模型库（如 `timm`），构建了 ViT 的核心组件，并在本地基于 CIFAR-10 数据集完成了简单的训练、评估以及注意力权重的提取与可视化工作。  
作为我的第一个作品，本项目会有一些疏漏之处，敬请批评指正。

## 📊 实验结果 

受本地算力限制，本项目只在小数据集上进行了初步的训练。基于后文的参数配置，本模型在 **CIFAR-10** 数据集上从零开始训练。经过 10 个 Epoch 的训练后，在 10000 张测试集图像上的最终分类准确率为：

> **模型测试集准确率：63.90%**

这一准确率并不如传统的CNN模型。这是由于Transformer 没有归纳偏置，在少量数据上表现较差。

训练过程中Loss变化如下图：

![Training Loss](<images/Training_Loss.png>)

此外，我提取了 Transformer 最后一层的 Attention 权重。以下是测试集中第一张图片的注意力分布热力图（不同颜色代表各个 Attention Head 关注的图像区域）：

![Attention Weight](<images/Attention_Weight.png>)

这表明模型的不同头确实在关注图像的不同区域。
## 🛠️ 实现细节

本项目在代码层面完成了以下模块的设计与实现：

* **Patch Embedding 分块投影**：
    * 使用 `nn.Conv2d` (kernel size 与 stride 均设为 patch_size) 实现图像的无重叠分块与特征维度映射。
* **Transformer Encoder 架构**：
    * 实现了标准的缩放点积多头自注意力机制 (Multi-Head Self-Attention)：
        $$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
* **ViT 主干网络集成**：
    * 实现了可学习的类别标记 (`[CLS] Token`) 与基于绝对位置的可学习位置编码 (`Positional Embedding`)。
    * 将输出的 `[CLS] Token` 特征接入全连接层完成最终的分类任务。
* **注意力权重提取与可视化**：
    * 在推断阶段，截取并导出了最后一个 Transformer Block 中 Attention 层的权重矩阵。
    * 针对 `[CLS] Token` 与各个 Patch 的注意力得分，将其重塑为二维空间网格，并生成各个注意力头的热力图，用于观察模型对输入图像的关注区域。

## ⚙️ 参数配置

### 1. 网络结构参数 
针对 32x32 的 CIFAR-10 图像，当前代码中默认的 ViT 初始化参数如下：

| 参数名 | 设定值 | 说明 |
| :--- | :--- | :--- |
| `img_size` | 32 | 输入图像尺寸 |
| `patch_size` | 4 | 图像分块尺寸 |
| `embed_dim` | 256 | 线性投影后的特征维度 |
| `head_num` | 8 | 多头注意力的头数 |
| `mlp_dim` | 512 | 前馈神经网络的隐藏层维度 |
| `num_layers` | 4 | Transformer Block 的堆叠层数 |
| `num_classes` | 10 | CIFAR-10 类别数 |

### 2. 训练超参数
在 `train.py` 中，模型训练的全局配置如下：

| 参数名 | 设定值 | 说明 |
| :--- | :--- | :--- |
| `BATCH_SIZE` | 64 | 数据加载批次大小 |
| `EPOCHS` | 10 | 训练总轮数 |
| `LR` (学习率) | 3e-4 | 初始学习率 |

损失函数为`CrossEntropyLoss`，优化器为`AdamW`，`weight_decay`为1e-2。

## 📂 项目结构

```text
├── images/                    # 存放 README 相关的展示图片
├── models/
│   ├── patch_embedding.py     # 负责图像分块与线性投影
│   ├── transformer_encoder.py # 包含 Attention 与 TransformerBlock
│   └── ViT.py                 # ViT 主干网络整合
├── train.py                   # 模型训练脚本
├── test.py                    # 模型测试与准确率评估
├── visualize.py               # 提取注意力权重并生成热力图
├── requirements.txt           # 项目依赖环境配置
└── README.md                  # 项目文档
```

## 🚀 运行指南

### 1. 环境准备
建议使用 Anaconda 创建独立的虚拟环境，然后通过 `requirements.txt` 一键安装所有依赖：
```bash
pip install -r requirements.txt
```

### 2. 训练模型
执行以下命令，程序将自动下载 CIFAR-10 数据集并启动训练。模型权重将在训练结束后自动保存为 `vit_cifar10.pth`：
```bash
python train.py
```

### 3. 测试评估
在测试集上评估模型的分类准确率：
```bash
python test.py
```

### 4. 运行可视化
执行以下脚本，将提取测试集第一张图片的注意力权重，并展示各个 Head 关注的区域热力图：
```bash
python visualize.py
```

## 🤔 思考与不足

作为一个纯粹的学习项目，这份从零手敲的基础版 ViT 虽然在代码逻辑上跑通了，但在实际性能表现上仍有很大的局限性。通过这次实践，我对 Transformer 架构在视觉领域的特性有了更深的体会：

**1. ViT 对小数据集的“水土不服”**

传统的 CNN 天生带有“平移不变性”和“局部相关性”这种强大的归纳偏置。而纯粹的 ViT 将图像直接打碎成无序的 Patch，它没有任何图像的先验知识，必须完全依靠海量数据从头开始“硬学”这些规律。
这导致**在 CIFAR-10 这种只有 5 万张训练集的小数据集上，ViT的表现不佳**，其实际准确率通常远低于同等参数规模的 CNN。

**2. 优化方向思考**

如果要提升模型在小数据集上的表现，接下来可以从以下几个方向优化：
- [ ] **强力的数据增强**：引入 `RandAugment`、`Mixup` 或 `CutMix`来扩充样本多样性。
- [ ] **更高级的训练策略**：原论文指出，ViT 对优化器和学习率极为敏感。加入 **学习率预热 (Warmup)** 和 **余弦退火调度器 (Cosine Annealing LR)** 会有更好的表现。

## 📚 参考资料

* **ViT 原论文**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ICLR 2021)
* **数据集**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
