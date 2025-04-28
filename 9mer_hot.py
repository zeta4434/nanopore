import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 数据加载与预处理
# -----------------------------
# 读取数据文件，文件中第一列为9mer序列，第二列为处理好的电流值
data = pd.read_csv("9mer_levels_v1_260bp.txt", header=None, sep="\t", names=["sequences", "values"])

# 将序列转换为大写（保证一致性，这里保留T）
data['sequences'] = data['sequences'].str.upper()

# 将电流值转换为数值，并删除缺失值
data['values'] = pd.to_numeric(data['values'], errors='coerce')
data = data.dropna()

# -----------------------------
# 独热编码：将9mer序列转换为特征
# -----------------------------
def one_hot_encode(seq):
    bases = ['A', 'G', 'C', 'T']
    # 对于9mer，每个位置生成4个特征，共计9*4=36列
    position_labels = [f'Position{i+1}_{base}' for i in range(9) for base in bases]
    encoded = pd.DataFrame(0, index=np.arange(1), columns=position_labels)
    for i, char in enumerate(seq):
        label = f'Position{i+1}_{char}'
        if label in encoded.columns:
            encoded.at[0, label] = 1
    return encoded

# 对所有序列进行独热编码
encoded_data = data['sequences'].apply(one_hot_encode)
X = pd.concat(encoded_data.tolist(), ignore_index=True)
y = data['values'].values  # 直接使用给定的电流值

# -----------------------------
# 划分数据集 & 训练线性回归模型
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 获取模型系数，每个特征对应一个系数
coef = model.coef_

# -----------------------------
# 生成每个位置上不同碱基间的预测电流变化差值，并绘制热力图
# -----------------------------
# 对于每个位置，计算“突变”前后电流预测差值: Δ = coef(突变后) - coef(突变前)
positions = [f'Position{i+1}' for i in range(9)]
bases = ['A', 'G', 'C', 'T']

# 统一设定颜色尺度（此处统一控制在 -0.4 到 0.4 之间）
vmin, vmax = -2, 2

# 创建一个3x3的子图布局（共9个位置）
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()  # 将子图数组展平，便于索引

for i, pos in enumerate(positions): 
    # 构建当前位置的 4x4 矩阵，行：原始碱基，列：突变后碱基
    heatmap_data = pd.DataFrame(index=bases, columns=bases)
    for base_from in bases:
        for base_to in bases:
            feature_from = f'{pos}_{base_from}'
            feature_to = f'{pos}_{base_to}'
            if feature_from in X.columns and feature_to in X.columns:
                # 计算预测电流变化差值：突变后碱基的贡献减去原始碱基的贡献
                delta = coef[X.columns.get_loc(feature_to)] - coef[X.columns.get_loc(feature_from)]
                heatmap_data.at[base_from, base_to] = delta

    # 绘制热力图，颜色尺度统一
    ax = axes[i]
    sns.heatmap(heatmap_data.astype(float), annot=True, cmap='coolwarm', fmt='.2f',
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Δ Predicted Value'}, ax=ax)
    ax.set_title(f'{pos} Predicted Current Change')
    ax.set_xlabel('From Nucleotide')
    ax.set_ylabel('To Nucleotide')

plt.tight_layout()
plt.show()
