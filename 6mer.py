import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
from itertools import combinations

# 设置 matplotlib 参数
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 确保能正确显示负号

# 读取数据
file_path = 'template_median68pA450bp.model'  # 替换为实际文件路径
data = pd.read_csv(file_path, sep="\t")

# 清洗与类型转换
data['level_mean'] = pd.to_numeric(data['level_mean'], errors='coerce')
data['kmer'] = data['kmer'].astype(str)
data = data.dropna(subset=['level_mean', 'kmer'])

# 独热编码函数
def one_hot_encode(seq):
    bases = ['A', 'G', 'C', 'T']
    cols = [f'Position{i+1}_{b}' for i in range(len(seq)) for b in bases]
    df = pd.DataFrame(0, index=[0], columns=cols)
    for i, nt in enumerate(seq):
        df.at[0, f'Position{i+1}_{nt}'] = 1
    return df

# 应用独热编码
encoded_list = data['kmer'].apply(one_hot_encode)
X = pd.concat(encoded_list.tolist(), ignore_index=True)

# 归一化目标变量
scaler = StandardScaler()
data['level_mean_norm'] = scaler.fit_transform(data[['level_mean']])
y = data['level_mean_norm'].values

# 保存预处理结果（可选）
data[['kmer', 'level_mean', 'level_mean_norm']].to_csv('output.csv', index=False)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 网格搜索参数
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# 测试集预测与指标
y_pred = best_model.predict(X_test)
print("MSE_test: ", mean_squared_error(y_test, y_pred))
print("MAE_test: ", mean_absolute_error(y_test, y_pred))
print("R2_test: ", r2_score(y_test, y_pred))

# 10 折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_mse = -cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_squared_error').mean()
cv_mae = -cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_absolute_error').mean()
cv_r2  =  cross_val_score(best_model, X, y, cv=kf, scoring='r2').mean()
print("\n10-Fold CV:")
print("CV MSE: ", cv_mse)
print("CV MAE: ", cv_mae)
print("CV R2: ", cv_r2)

# 特征重要性
imp = best_model.get_booster().get_score(importance_type='weight')
imp_df = pd.DataFrame({
    'feature': list(imp.keys()),
    'importance': list(imp.values())
}).sort_values('importance', ascending=False)
imp_df.plot(kind='bar', x='feature', y='importance', title='Feature Importance', legend=False)
plt.tight_layout()
plt.show()

# 合并全数据用于 SHAP
X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = np.concatenate([y_train, y_test])

# SHAP 分析
explainer = shap.TreeExplainer(best_model)
shap_values_train = explainer.shap_values(X_train)
shap_values_test  = explainer.shap_values(X_test)
shap_inter_train  = explainer.shap_interaction_values(X_train)
shap_inter_test   = explainer.shap_interaction_values(X_test)

# 初始化 JS（在 Notebook/浏览器中可互动）
shap.initjs()

# 1. SHAP Summary Plots
shap.summary_plot(shap_values_train, X_train, plot_type="bar", show=True)
shap.summary_plot(shap_values_train, X_train, show=True)
shap.summary_plot(shap_values_test,  X_test,  plot_type="bar", show=True)
shap.summary_plot(shap_values_test,  X_test,  show=True)

# 2. Force Plots（力图）示例：前三个测试样本
for i in range(3):
    shap.force_plot(
        explainer.expected_value,
        shap_values_test[i],
        X_test.iloc[i],
        matplotlib=True,
        show=True
    )

# 3. Waterfall Plot（瀑布图）示例：第一个测试样本
shap.plots.waterfall(
    shap.Explanation(values=shap_values_test[0],
                     base_values=explainer.expected_value,
                     data=X_test.iloc[0]),
    show=True
)

# 4. Interaction Plots（交互图）
# 指定一对感兴趣的特征，例如 Position5_A 与 Position10_T
pair = ('Position3_T', 'Position3_C')
shap.dependence_plot(
    pair,
    shap_inter_test,
    X_test,
    interaction_index=None,
    show=True
)

# （可选）批量生成更多交互图
pairs = [
    ('Position3_G', 'Position4_C'),
    ('Position3_C', 'Position4_T'),
    # 根据实际情况添加
]
for p in pairs:
    shap.dependence_plot(
        p,
        shap_inter_test,
        X_test,
        interaction_index=None,
        show=True
    )

# 5. 残差图
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_test, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()
