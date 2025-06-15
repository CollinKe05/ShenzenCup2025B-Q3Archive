import numpy as np
from skimage import color

def rgb_to_lab(rgb_array):
    """
    将RGB数组转换为Lab颜色空间。
    参数:
        rgb_array: 形状为(N, 3)的RGB数组，值范围为0-255。
    返回:
        lab_array: 形状为(N, 3)的Lab数组。
    """
    # 归一化到0-1
    rgb_normalized = rgb_array / 255.0
    # 转换到Lab
    lab_array = color.rgb2lab(rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
    return lab_array
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def calculate_delta_e00(lab1, lab2):
    """
    计算两个Lab颜色之间的ΔE00差异。
    参数:
        lab1, lab2: 形状为(N, 3)的Lab数组。
    返回:
        delta_e: 形状为(N,)的ΔE00差异数组。
    """
    delta_e = []
    for l1, l2 in zip(lab1, lab2):
        color1 = LabColor(*l1)
        color2 = LabColor(*l2)
        delta = delta_e_cie2000(color1, color2)
        delta_e.append(delta)
    return np.array(delta_e)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def train_mlp_regressor(X, y):
    """
    训练MLP回归模型。
    参数:
        X: 形状为(N, 3)的输入RGB数组。
        y: 形状为(N, 3)的目标Lab数组。
    返回:
        model: 训练好的MLP回归模型。
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=500)
    model.fit(X_train, y_train)
    return model
def correct_colors(rgb_array, model, threshold=2.3):
    """
    对RGB数组进行颜色校正。
    参数:
        rgb_array: 形状为(N, 3)的RGB数组。
        model: 训练好的MLP回归模型。
        threshold: ΔE00的阈值，低于该值认为颜色差异可接受。
    返回:
        corrected_rgb: 形状为(N, 3)的校正后RGB数组。
    """
    lab_array = rgb_to_lab(rgb_array)
    # 假设目标Lab为某一标准值，这里以(50, 0, 0)为例
    target_lab = np.array([50, 0, 0])
    target_lab_array = np.tile(target_lab, (lab_array.shape[0], 1))
    delta_e = calculate_delta_e00(lab_array, target_lab_array)
    # 找出需要校正的索引
    indices_to_correct = np.where(delta_e &gt; threshold)[0]
    # 对需要校正的颜色进行预测
    if len(indices_to_correct) &gt; 0:
        rgb_to_correct = rgb_array[indices_to_correct]
        lab_predicted = model.predict(rgb_to_correct)
        # 将预测的Lab转换回RGB
        lab_predicted_reshaped = lab_predicted.reshape(1, -1, 3)
        rgb_corrected = color.lab2rgb(lab_predicted_reshaped).reshape(-1, 3)
        rgb_corrected = np.clip(rgb_corrected * 255, 0, 255).astype(np.uint8)
        # 替换原始RGB值
        rgb_array[indices_to_correct] = rgb_corrected
    return rgb_array
'''
# 示例RGB数据，形状为(N, 3)
rgb_data = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)

# 转换到Lab颜色空间
lab_data = rgb_to_lab(rgb_data)

# 训练MLP回归模型
mlp_model = train_mlp_regressor(rgb_data, lab_data)

# 进行颜色校正
corrected_rgb = correct_colors(rgb_data.copy(), mlp_model)

# 可视化校正前后的颜色差异
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow([[rgb_data[i] / 255.0]])
    plt.axis('off')
    plt.subplot(2, 10, i + 11)
    plt.imshow([[corrected_rgb[i] / 255.0]])
    plt.axis('off')
plt.suptitle('Top: Original Colors, Bottom: Corrected Colors')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from sklearn.neural_network import MLPRegressor
from scipy.spatial import Delaunay
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# 模拟64个像素，每个像素是一个3x3的RGB矩阵
np.random.seed(0)
pixels = np.random.randint(0, 256, (64, 3, 3, 3), dtype=np.uint8)

# 提取每个像素中心点的RGB值，并归一化
center_rgbs = np.array([pixel[1, 1] / 255.0 for pixel in pixels])

# 转换为Lab颜色空间
center_labs = rgb2lab(center_rgbs.reshape(1, -1, 3)).reshape(-1, 3)

# 构建目标色域：100个Lab点，组成凸包
target_lab = np.random.uniform([20, -40, -40], [80, 40, 40], (100, 3))
hull = Delaunay(target_lab)

# 判断每个Lab是否在凸包内（目标色域内）
inside = hull.find_simplex(center_labs) &gt;= 0

# 构建MLP训练集（仅训练目标色域外的点）
X_train = center_labs[~inside]
Y_train = np.array([    target_lab[np.argmin(np.linalg.norm(target_lab - lab, axis=1))]
    for lab in X_train
])

# 使用MLP拟合映射关系
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, Y_train)

# 对目标色域外的点使用MLP进行映射
mapped_labs = center_labs.copy()
mapped_labs[~inside] = mlp.predict(center_labs[~inside])

# 计算ΔE00色差
def compute_deltaE00_batch(lab1_array, lab2_array):
    return np.array([        delta_e_cie2000(LabColor(*lab1), LabColor(*lab2))
        for lab1, lab2 in zip(lab1_array, lab2_array)
    ])

delta_E = compute_deltaE00_batch(center_labs, mapped_labs)
mean_delta_E = np.mean(delta_E)
std_delta_E = np.std(delta_E)

# 绘制ΔE00直方图
plt.hist(delta_E, bins=20, color='skyblue', edgecolor='black')
plt.title(f'ΔE00 Histogram\nMean: {mean_delta_E:.2f}, Std: {std_delta_E:.2f}')
plt.xlabel('ΔE00')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
'''
