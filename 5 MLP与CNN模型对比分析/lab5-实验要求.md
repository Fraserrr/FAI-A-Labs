# lab5：MLP与CNN模型对比分析

[TOC]

## 实验概述

本实验旨在通过对多层感知机（MLP）和卷积神经网络（CNN）的实现、训练和评估，帮助学生深入理解两种模型的结构特点、性能差异以及适用场景。学生将从基础模型开始，逐步探索更复杂的网络架构，最终通过对比分析，掌握深度学习模型设计与评估的关键技能。

本实验的代码已经可以稳定运行。作业内容包括补全两个模型定义代码（MLP与CNN）以及回答一系列问题。两个补全任务的代码仅需在实验报告中体现即可。



## 实验目的

1. 掌握MLP和CNN的基本原理和实现方法
2. 了解不同网络结构对模型性能的影响
3. 学习深度学习模型训练、评估和可视化的方法
4. 通过对比实验，理解不同模型在图像分类任务中的优缺点
5. 培养深度学习模型调优和问题解决的能力



## 实验准备

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy, Matplotlib
- scikit-learn (用于评估)
- 建议使用GPU环境（可选）

实验环境已经在mo平台中搭建好了，同学们无需自行配置

### 实验数据集

本实验使用CIFAR-10数据集，包含10个类别的彩色图像，每类6000张，共60000张32×32的图像。

### 项目结构

```
项目根目录/
├── models/
│   ├── __init__.py
│   ├── mlp.py        # MLP模型定义
│   └── cnn.py        # CNN模型定义
├── utils/
│   ├── __init__.py
│   ├── data_loader.py  # 数据加载函数
│   └── train_utils.py  # 训练和评估函数
├── train_all_notebook.py    # 统一训练脚本
└── compare_models.py        # 模型比较脚本
```



## 实验原理

### 多层感知机(MLP)

多层感知机是一种前馈神经网络，由输入层、一个或多个隐藏层和输出层组成。MLP的主要特点是：

1. 每层神经元与下一层全连接
2. 使用非线性激活函数（如ReLU、Sigmoid等）
3. 通过反向传播算法进行训练

**思考问题1**: MLP在处理图像数据时面临哪些挑战？请从数据结构、参数量和特征提取能力三个角度分析。


### 卷积神经网络(CNN)

卷积神经网络是为处理具有网格状拓扑结构的数据而设计的神经网络，主要包含卷积层、池化层和全连接层。CNN的主要特点是：

1. 局部连接：每个神经元只与输入数据的一个局部区域连接
2. 权重共享：同一特征图的所有神经元共享相同的权重
3. 多层次特征提取：低层检测边缘等简单特征，高层组合这些特征形成更复杂的表示

**思考问题2**: CNN相比MLP在处理图像时具有哪些优势？解释卷积操作如何保留图像的空间信息。



## 实验内容

### 第一部分：基础MLP模型

#### 1.1 了解MLP模型结构

查看`models/mlp.py`文件，理解三种MLP模型的结构：
- `SimpleMLP`: 单隐层MLP
- `DeepMLP`: 多隐层MLP，带有BatchNorm和Dropout
- `ResidualMLP`: 带有残差连接的MLP

**任务1**: 在下面的代码块中，实现一个具有两个隐藏层的MLP模型。第一隐藏层有128个神经元，第二隐藏层有64个神经元，输出层对应10个类别。使用ReLU激活函数，并添加BatchNorm和Dropout(0.3)。

```python
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=3*32*32):
        super(TwoLayerMLP, self).__init__()
        self.flatten = nn.Flatten()      
        # 使用nn.Linear, nn.BatchNorm1d, nn.ReLU和nn.Dropout实现两个隐藏层
        
    def forward(self, x):
        x = self.flatten(x)      
        # 实现前向传播
        return x
```

#### 1.2 训练和评估MLP模型

1. 在 `train.ipynb` 中训练SimpleMLP模型，确保将`model_type`设置为`'simple_mlp'`。

2. 观察训练过程中的损失和准确率变化，以及最终在测试集上的性能。

    **分析问题1**: 训练过程中，损失和准确率曲线表现如何？是否出现过拟合或欠拟合？简要分析可能的原因。


3. 修改参数尝试训练DeepMLP模型，将`model_type`设置为`'deep_mlp'`。

    **分析问题2**: 对比SimpleMLP和DeepMLP的性能，增加网络深度对性能有何影响？



### 第二部分：基础CNN模型

#### 2.1 了解CNN模型结构

查看`models/cnn.py`文件，理解不同CNN模型的结构：
- `SimpleCNN`: 简单的CNN，包含两个卷积层
- `MediumCNN`: 中等复杂度的CNN，带有BatchNorm和Dropout
- `VGGStyleNet`: VGG风格的CNN，使用连续的3x3卷积
- `SimpleResNet`: 简化的ResNet，包含残差连接

**任务2**: 修改下面的`SimpleCNN`代码，添加一个额外的卷积层和BatchNorm。新的卷积层应该在第二个池化层之后，卷积核数量为64，卷积核大小为3x3。

```python
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 在这里添加一个新的卷积层、BatchNorm和相应的池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # 修改全连接层以适应新的特征图尺寸
        self.relu = nn.ReLU()
    def forward(self, x):
        # 实现包含新卷积层的前向传播
        return x
```

#### 2.2 训练和评估CNN模型

1. 在 `train.ipynb` 中训练SimpleMLP模型，确保将`model_type`设置为`'simple_cnn'`，并将`use_data_augmentation`设置为`True`。

2. 观察训练过程和卷积核可视化结果。

    **分析问题3**: 卷积核可视化显示了什么模式？这些模式与图像中的哪些特征可能对应？


3. 继续训练MediumCNN模型，将`model_type`设置为`'medium_cnn'`。

    **分析问题4**: CNN模型相比MLP在CIFAR-10上的性能有何不同？为什么会有这样的差异？



### 第三部分：高级CNN架构探索

#### 3.1 VGG风格和ResNet风格网络架构

在本部分中，我们将探索两种影响深远的CNN架构：VGG和ResNet。通过理解这些经典架构的设计理念和特点，可以帮助我们设计更高效的神经网络。

##### 3.1.1 VGG架构特点
VGG网络（由Visual Geometry Group开发）是一种非常简洁而有效的CNN架构，在2014年ImageNet挑战赛中取得了优异成绩。其主要特点包括：

1. **简单统一的设计**：使用小尺寸（3×3）卷积核和2×2最大池化层
2. **深度堆叠**：通过堆叠多个相同配置的卷积层增加网络深度
3. **结构规整**：遵循"卷积层组-池化层"的模式，随着网络深入，特征图尺寸减小而通道数增加

在我们的实现中，`VGGStyleNet`采用了简化版的VGG设计理念，包含三个卷积块，每个块包含两个卷积层和一个池化层。

1. 在 `train.ipynb` 中训练SimpleMLP模型，确保将`model_type`设置为`'vgg_style'`，并将`use_data_augmentation`设置为`True`。

2. 观察网络的训练过程和性能。特别注意其收敛速度和最终准确率。

##### 3.1.2 ResNet架构及残差连接

ResNet（残差网络）由微软研究院的He等人在2015年提出，是解决"深度退化问题"的突破性架构。其核心创新是引入了残差连接（skip connection）：

1. **残差连接**：通过快捷连接（shortcut connection）将输入直接加到输出上，形成恒等映射路径
2. **残差学习**：网络不再直接学习输入到输出的映射F(x)，而是学习残差F(x)-x
3. **深度扩展**：残差连接有效缓解了梯度消失问题，使得训练非常深的网络成为可能

在我们的实现中，`SimpleResNet`使用了基本的残差块，每个残差块包含两个3×3的卷积层和一个跳跃连接。

1. 在 `train.ipynb` 中训练SimpleMLP模型，确保将`model_type`设置为`'resnet'`，并将`use_data_augmentation`设置为`True`。

2. 观察网络的训练过程和性能，特别是深度对训练稳定性的影响。

##### 3.1.3 Bottleneck结构

在更深的ResNet变体中，常使用"瓶颈"（Bottleneck）结构来降低计算复杂度：

- 使用1×1卷积降低通道数（降维）
- 使用3×3卷积进行特征提取
- 再使用1×1卷积恢复通道数（升维）

这种设计大幅减少参数量和计算量，同时保持或提高性能。

**思考问题3**: 分析Bottleneck结构的优势。为什么1×1卷积在深度CNN中如此重要？它如何帮助控制网络的参数量和计算复杂度？

**探索问题1**: 查看`models/cnn.py`中的`SimpleResNet`实现，分析残差连接是如何实现的。如果输入和输出通道数不匹配，代码是如何处理的？

#### 3.2 模型复杂度分析

不同CNN架构在性能和效率之间存在权衡。现在我们将通过分析不同模型的参数量和推理时间来理解这种权衡。

1. 运行以下代码来分析各个模型的复杂度：
   ```python
   from models import SimpleMLP, DeepMLP, ResidualMLP, SimpleCNN, MediumCNN, VGGStyleNet, SimpleResNet
   from utils import model_complexity
   import torch
   
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
   models = {
       'SimpleMLP': SimpleMLP(),
       'DeepMLP': DeepMLP(),
       'SimpleCNN': SimpleCNN(),
       'MediumCNN': MediumCNN(),
       'VGGStyleNet': VGGStyleNet(),
       'SimpleResNet': SimpleResNet()
   }
   
   results = {}
   for name, model in models.items():
       print(f"\n分析{name}复杂度:")
       params, time = model_complexity(model, device=device)
       results[name] = {'params': params, 'time': time}
   ```

2. 记录并比较各个模型的参数量和推理时间。

**分析问题5**: VGG风格和ResNet风格网络的性能比较。残差连接带来了哪些优势？

**分析问题6**: 参数量和推理时间如何影响模型的实用性？如何在性能和效率之间找到平衡？


#### 3.3 理解高级CNN设计理念

随着深度学习的发展，CNN架构设计也变得更加精细和高效。以下是一些重要的设计理念：

1. **网络深度与宽度平衡**：更深的网络能学习更抽象的特征，但也更难训练；更宽的网络（更多通道）能捕获更多特征，但参数量增加
2. **跳跃连接**：除了ResNet的残差连接，还有DenseNet的密集连接、U-Net的跨层连接等
3. **特征增强**：注意力机制（如SENet的通道注意力）、特征融合等
4. **高效卷积设计**：深度可分离卷积（MobileNet）、组卷积（ShuffleNet）等

**探索问题2**: 如果你要为移动设备设计一个CNN模型，应该考虑哪些因素来权衡性能和效率？请提出至少三条具体的设计原则。



### 第四部分：模型比较与分析

运行 `compare.py` 来对比不同模型的性能：

**综合分析**: 根据比较结果，分析不同类型模型（MLP和CNN）以及不同复杂度模型的性能差异。考虑以下几点：
1. 测试准确率
2. 参数量
3. 推理时间
4. 训练收敛速度
5. 过拟合/欠拟合情况

下面我将提供CNN手写数字识别的pytorch和tensorflow的代码，请你运行跑通两段代码并感受二者的不行（附上截图）：
# pytorch版本：
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

    # 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 对于灰度图像进行归一化
])

    # 下载并加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输出形状: (6, 26, 26)
        self.pool = nn.MaxPool2d(2, 2)   # 池化后: (6, 13, 13)
        self.conv2 = nn.Conv2d(6, 16, 3) # 输出形状: (16, 11, 11)
        # 池化后: (16, 5, 5) → 展平为16*5*5=400
        self.fc1 = nn.Linear(16 * 5 * 5, 100)  # 修正输入特征数
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 确保展平为400
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    # 设置模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
```

# tensorflow版本
```python
import tensorflow as tf
from tensorflow.keras import layers, datasets, models

    # 数据预处理
def preprocess_data():
    # 加载数据集
    (train_images, train_labels), (_, _) = datasets.mnist.load_data()
    
    # 添加通道维度并归一化到 [-1, 1]
    train_images = train_images[..., tf.newaxis]  # 形状从 (60000, 28,28) 变为 (60000, 28,28,1)
    train_images = (train_images.astype('float32') / 127.5) - 1.0
    
    # 转换为 Dataset 对象
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(60000).batch(32)
    return train_dataset
    
    # 创建 CNN 模型
def create_model():
    model = models.Sequential([
        # 卷积部分
        layers.Conv2D(6, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(2),
        
        # 全连接部分
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax')  # 输出层使用 softmax
    ])
    return model
    
    # 训练配置
def train_model():
    # 获取数据
    train_dataset = preprocess_data()
    
    # 创建模型
    model = create_model()
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    
    # 训练模型
    model.fit(
        train_dataset,
        epochs=10,
        verbose=1)
    
    # 执行训练
if __name__ == "__main__":
    train_model()
```

## 创新探索任务（选做）

选择下列一项或多项任务完成：

1. **模型改进**：对任一模型进行修改和改进，提高其在CIFAR-10上的性能。
2. **可视化分析**：设计更好的可视化方法来解释模型的决策过程。
3. **迁移学习**：探索如何利用预训练模型提高CIFAR-10的分类性能。
4. **对抗性样本**：生成对抗性样本，并研究不同模型对对抗性样本的鲁棒性。
5. **自监督学习**：实现一个简单的自监督学习方法，并评估其效果。



## 实验报告要求

实验报告应包含以下内容：

1. 实验目的和背景介绍
2. 实验原理简述
3. 实验过程描述
4. 实现的代码（关键部分，包含详细注释）
5. 实验结果和分析（包括填写的所有分析问题和任务）
6. 创新探索任务的设计、实现和结果（如果选做）
7. 结论和思考
8. 参考文献



## 评分标准

- 基础任务完成度：60%
- 分析问题深度和准确性：35%
- 创新探索任务：15% (bonus)
- 报告质量和表达清晰度：5%



## 参考资料

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
4. PyTorch文档：https://pytorch.org/docs/stable/index.html
5. CS231n: Convolutional Neural Networks for Visual Recognition：https://cs231n.github.io/