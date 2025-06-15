import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from colour import XYZ_to_Lab, RGB_to_XYZ
from colour.difference import delta_E_CIE2000
import os
import matplotlib.pyplot as plt


# ================= 增强数据预处理 =================
def load_data_with_neighbors(data_dir):
    """加载数据并构建包含邻域信息的5D张量"""
    channels = ['R', 'G', 'B']
    tensor_4d = np.zeros((64, 64, 3, 3))  # [H, W, LED_type, RGB]

    # 加载原始数据
    for led_idx, led in enumerate(channels):
        for color_idx, color in enumerate(channels):
            filename = f"{led}_{color}.csv"
            path = os.path.join(data_dir, filename)
            df = pd.read_csv(path, header=None)
            tensor_4d[:, :, led_idx, color_idx] = df.values

    # 添加边界填充并构建邻域信息
    padded = np.pad(tensor_4d / 255.0, ((1, 1), (1, 1), (0, 0), (0, 0)), mode='edge')

    # 构建包含邻域信息的5D张量 [H, W, LED_type, RGB, Neighbors]
    tensor_5d = np.zeros((64, 64, 3, 3, 9))
    for i in range(64):
        for j in range(64):
            # 获取3x3邻域 (包括自身)
            neighborhood = padded[i:i + 3, j:j + 3]
            tensor_5d[i, j] = neighborhood.reshape(3, 3, 9)

    return tensor_5d


# ================= 增强数据集定义 =================
class LEDDatasetWithNeighbors(Dataset):
    def __init__(self, data_tensor, target=0.86):
        self.data = data_tensor  # [H, W, LED_type, RGB, 9]
        self.target = torch.full((3,), target, dtype=torch.float32)

    def __len__(self):
        return 64 * 64

    def __getitem__(self, idx):
        i, j = idx // 64, idx % 64
        # 当前像素数据 + 邻域信息
        pixel_data = torch.tensor(self.data[i, j], dtype=torch.float32)
        return pixel_data, self.target


# ================= 增强模型定义 =================
class EnhancedCorrectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, channels=9, height=3, width=3]
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc_net(x)


# ================= 能量最小化损失函数 =================
class EnergyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # ΔE2000权重
        self.beta = beta  # MSE权重
        self.gamma = gamma  # 空间一致性权重

    def forward(self, pred, target, outputs_map):
        """
        pred: 当前像素预测值 [batch, 3]
        target: 目标值 [3]
        outputs_map: 完整输出图 [64, 64, 3]
        """
        # ΔE2000损失
        deltaE = delta_E_CIE2000(
            XYZ_to_Lab(RGB_to_XYZ(pred.detach().numpy().reshape(1, 3))),
            XYZ_to_Lab(RGB_to_XYZ(target.numpy().reshape(1, 3))))
        deltaE_loss = torch.tensor(deltaE).mean()

        # MSE损失
        mse_loss = nn.MSELoss()(pred, target)

        # 空间一致性损失
        spatial_loss = 0.0
        if outputs_map is not None:
        # 计算水平和垂直梯度
            grad_x = outputs_map[:, 1:, :] - outputs_map[:, :-1, :]
        grad_y = outputs_map[1:, :, :] - outputs_map[:-1, :, :]
        spatial_loss = (grad_x.pow(2).mean() + grad_y.pow(2).mean())

        return (
                self.alpha * deltaE_loss +
                self.beta * mse_loss +
                self.gamma * spatial_loss
        )


# ================= 增强训练流程 =================
def enhanced_train(data_dir, epochs=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载增强数据
    tensor_5d = load_data_with_neighbors(data_dir)
    dataset = LEDDatasetWithNeighbors(tensor_5d)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化模型
    model = EnhancedCorrectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 可视化准备
    plt.figure(figsize=(12, 6))
    loss_history = []

    # 能量最小化训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        full_output = torch.zeros((64, 64, 3))  # 存储完整输出图

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs.permute(0, 3, 1, 2))  # [batch, 9, 3, 3] → [batch, 3]

            # 更新完整输出图
            indices = [(i, j) for i in range(64) for j in range(64)][batch_idx * 64: (batch_idx + 1) * 64]
            for k, (i, j) in enumerate(indices):
                full_output[i, j] = outputs[k].detach()

            # 计算损失
            loss_fn = EnergyLoss()
            loss = loss_fn(outputs, targets, full_output if batch_idx > 0 else None)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录并可视化
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        # 实时可视化
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(full_output.detach().numpy())
        plt.title(f"Corrected Output (Epoch {epoch + 1})")

        plt.subplot(1, 2, 2)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.pause(0.1)

    plt.show()
    return model


# ================= 主程序 =================
if __name__ == "__main__":
    data_dir = "D:/Pyproject/CSV"

    # 训练增强模型
    model = enhanced_train(data_dir, epochs=50)

    # 保存最终模型
    torch.save(model.state_dict(), "enhanced_led_correction.pth")

    # 示例应用
    test_input = torch.randn(9, 3, 3)  # 模拟邻域输入
    corrected = model(test_input.unsqueeze(0))
    print(f"Input shape: {test_input.shape}")
    print(f"Corrected RGB: {corrected.detach().numpy()}")
