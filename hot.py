import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import seaborn as sns

# 设置matplotlib的参数
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 确保能正确显示负号

# 读取template_median68pA.model文件（你的示例文件）
file_path = 'template_median68pA450bp.model'  # 替换为实际文件路径
data = pd.read_csv(file_path, sep="\t")

# 确保数据类型正确，并清理掉无效行
data['level_mean'] = pd.to_numeric(data['level_mean'], errors='coerce')
data['kmer'] = data['kmer'].astype(str)

# 删除包含 NaN 的行
data = data.dropna()

# 数据预处理：将序列（kmer）转换为独热编码
def one_hot_encode(seq):
    bases = ['A', 'G', 'C', 'T']
    position_labels = [f'Position{i+1}_{base}' for i in range(len(seq)) for base in bases]
    encoded = pd.DataFrame(0, index=np.arange(1), columns=position_labels)
    for i, char in enumerate(seq):
        label = f'Position{i+1}_{char}'
        encoded.at[0, label] = 1
    return encoded

# 应用独热编码
encoded_data = data['kmer'].apply(one_hot_encode)
X = pd.concat(encoded_data.tolist(), ignore_index=True)

# 使用归一化对level_mean列进行处理
scaler_level_mean = StandardScaler()
data['level_mean_normalized'] = scaler_level_mean.fit_transform(data[['level_mean']])

# 选用归一化后的level_mean作为目标变量
y = data['level_mean_normalized'].values

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置 XGBoost 参数进行网格搜索
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300]
}

model = XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# 训练模型，使用归一化后的 level_mean 作为目标变量
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 输出最佳模型参数
print("best_parameter：", grid_search.best_params_)

# 使用最佳模型进行预测
y_pred = best_model.predict(X_test)

# 计算并输出性能指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE_test:", mse)
print("MAE_test:", mae)
print("R_square_test:", r2)

# SHAP分析部分
explainer = shap.TreeExplainer(best_model)

# 计算SHAP值
shap_values_train = explainer.shap_values(X_train)
shap_values_test = explainer.shap_values(X_test)

# 将SHAP值转换为DataFrame，行表示样本，列表示特征
shap_values_train_df = pd.DataFrame(shap_values_train, columns=X_train.columns)
shap_values_test_df = pd.DataFrame(shap_values_test, columns=X_test.columns)

# 对每个位置的SHAP值进行归一化处理，以0为均值
shap_values_normalized = shap_values_train_df - shap_values_train_df.mean(axis=0)

# 获取每个位置的特征列
positions = [f'Position{i+1}' for i in range(6)]
bases = ['A', 'G', 'C', 'T']

# 手动设置一致的色标范围
vmin = -0.4
vmax = 0.4

# 创建子图：2行3列的布局
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()  # 将axes数组展平，方便索引

# 针对每个位置绘制一个热力图
for i, position in enumerate(positions):
    ax = axes[i]  # 获取当前子图
    heatmap_data = pd.DataFrame(index=bases, columns=bases)

    for base_1 in bases:
        for base_2 in bases:
            feature_1 = f'{position}_{base_1}'
            feature_2 = f'{position}_{base_2}'
            if feature_1 in shap_values_normalized.columns and feature_2 in shap_values_normalized.columns:
                # 计算当前特征的平均绝对SHAP值
                mean_abs_shap = shap_values_normalized[feature_1].abs().mean() - shap_values_normalized[feature_2].abs().mean()
                heatmap_data.at[base_1, base_2] = mean_abs_shap

    # 绘制热力图，并确保color scale一致
    sns.heatmap(heatmap_data.astype(float), annot=True, cmap='coolwarm', fmt='.2f', vmin=vmin, vmax=vmax, 
                cbar_kws={'label': 'Mean Abs SHAP Value'}, ax=ax)
    ax.set_title(f'Mean Absolute SHAP Values for Position {i+1}')
    ax.set_xlabel('Base Change')
    ax.set_ylabel('Base Change')

# 调整布局，使得子图不重叠
plt.tight_layout()
plt.show()
