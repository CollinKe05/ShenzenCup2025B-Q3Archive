import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from colour import XYZ_to_Lab, RGB_to_XYZ
from colour.difference import delta_E_CIE2000
import os


# ================= 数据预处理部分 =================
def load_data(data_dir):
    """加载所有CSV文件并构建3D张量"""
    channels = ['R', 'G', 'B']
    tensor_3d = np.zeros((64, 64, 3, 3))  # [H, W, LED_type, RGB]

    for led_idx, led in enumerate(channels):
        for color_idx, color in enumerate(channels):
            filename = f"{led}_{color}.csv"
            path = os.path.join(data_dir, filename)
            df = pd.read_csv(path, header=None)
            tensor_3d[:, :, led_idx, color_idx] = df.values

    return tensor_3d / 255.0  # 归一化到0-1


# ================= 数据集定义 =================
class LEDDataset(Dataset):
    def __init__(self, data_tensor, target=0.86):  # 220/255≈0.86
        self.data = data_tensor
        self.target = torch.full((3,), target, dtype=torch.float32)

    def __len__(self):
        return 64 * 64

    def __getitem__(self, idx):
        i, j = idx // 64, idx % 64
        pixel_data = torch.tensor(self.data[i, j], dtype=torch.float32)
        return pixel_data, self.target


# ================= 模型定义 =================
class CorrectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # 输出0-1范围
        )

    def forward(self, x):
        return self.net(x.view(-1, 9))


# ================= 训练函数 =================
def train_model(data_tensor, epochs=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据
    dataset = LEDDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化模型和优化器
    model = CorrectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 混合损失函数
    def combined_loss(pred, target, rgb_data):
        # MSE损失
        mse_loss = nn.MSELoss()(pred, target)

        # ΔE00损失
        with torch.no_grad():
            # 转换到Lab空间
            xyz = RGB_to_XYZ(pred.cpu().numpy().reshape(1, 3))
            lab_pred = XYZ_to_Lab(xyz)

            xyz_target = RGB_to_XYZ(target.cpu().numpy().reshape(1, 3))
            lab_target = XYZ_to_Lab(xyz_target)

        deltaE = delta_E_CIE2000(lab_pred, lab_target)
        deltaE_loss = torch.tensor(deltaE, dtype=torch.float32)

        return 0.7 * deltaE_loss + 0.3 * mse_loss

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = combined_loss(outputs, targets, inputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs} Loss: {total_loss / len(loader):.4f}')

    return model


# ================= 主程序 =================
if __name__ == "__main__":
    # 加载数据
    data_dir = "D:/Pyproject/CSV"
    display_data = load_data(data_dir)

    # 训练校正模型
    model = train_model(display_data)

    # 保存模型
    torch.save(model.state_dict(), "led_correction_model.pth")

    # 应用校正（示例）
    test_pixel = torch.tensor(display_data[32, 32], dtype=torch.float32)
    corrected = model(test_pixel)
    print(f"原始值: {test_pixel.numpy()}")
    print(f"校正值: {corrected.detach().numpy()}")
