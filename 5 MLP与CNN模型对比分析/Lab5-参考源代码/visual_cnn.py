import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from models import SimpleMLP, DeepMLP, ResidualMLP, SimpleCNN, MediumCNN, VGGStyleNet, SimpleResNet, ImprovedResNet

# 配置参数
SAVE_DIR = "model_visualization_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 可视化配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=20, shuffle=True)


# Grad-CAM实现
def grad_cam(model, input_tensor, target_class):
    model.eval()
    target_layer = model.conv2
    activations = None
    gradients = None

    # 前向hook
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    # 反向hook
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)

    # 前向传播
    output = model(input_tensor.unsqueeze(0))
    model.zero_grad()
    output[0, target_class].backward()

    # 计算权重
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    heatmap = torch.mean(activations * pooled_gradients[..., None, None], dim=1)
    heatmap = np.maximum(heatmap.squeeze().cpu().numpy(), 0)
    heatmap /= heatmap.max()

    hook_f.remove()
    hook_b.remove()
    return heatmap


# 组合可视化
def create_visualization(model, img, label, save_path):
    fig = plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(231)
    img_show = img.numpy().transpose(1, 2, 0)
    img_show = (img_show * 0.5) + 0.5  # 反归一化
    plt.imshow(img_show)
    plt.title(f'Input Image\nClass: {label}')
    plt.axis('off')

    # Grad-CAM
    plt.subplot(232)
    heatmap = grad_cam(model, img, label)
    plt.imshow(img_show)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM')
    plt.axis('off')

    # 显著图
    plt.subplot(233)
    img.requires_grad = True
    output = model(img.unsqueeze(0))
    output[0, label].backward()
    saliency = img.grad.data.abs().max(dim=1)[0].squeeze().numpy()
    plt.imshow(saliency, cmap='hot')
    plt.title('Saliency Map')
    plt.axis('off')

    # 特征空间投影（修正后的特征提取）
    plt.subplot(234)
    features, labels_list = [], []
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(testloader):
            # 正确的前向传播流程
            x = model.conv1(imgs)
            x = model.relu(x)
            x = model.pool(x)
            x = model.conv2(x)
            x = model.relu(x)
            x = model.pool(x)

            features.append(x.view(x.size(0), -1))
            labels_list.append(targets)
            if i >= 20:  # 限制样本数量
                break

    features = torch.cat(features).numpy()
    labels = torch.cat(labels_list).numpy()

    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(features[:500])  # 限制计算量

    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels[:500],
                cmap=plt.cm.tab10, alpha=0.6, s=10)
    plt.colorbar(ticks=range(10))
    plt.title('Feature Projection')

    # 遮挡分析
    plt.subplot(235)
    occlusion = Occlusion(model)
    attribution = occlusion.attribute(
        img.unsqueeze(0),
        strides=(3, 8, 8),
        target=label,
        sliding_window_shapes=(3, 15, 15),
        baselines=0
    )
    plt.imshow(attribution.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    plt.title('Occlusion Analysis')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# 安全加载模型
def load_model(model_path):
    return torch.load(model_path, map_location='cpu', weights_only=True)


if __name__ == "__main__":
    # 初始化模型并加载参数
    model = SimpleCNN()
    model.load_state_dict(load_model('ck/SimpleCNN_best.pth'))

    # 获取测试样本
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # 生成可视化
    for idx in range(10):
        img = images[idx]
        label = labels[idx].item()
        save_path = os.path.join(SAVE_DIR, f'cnn_visual_{idx}.png')
        create_visualization(model, img, label, save_path)
