import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False})

DATA_PATH = "9mer_levels_v1_400bp.txt"  # 数据文件路径

# -----------------------------
# 1. 数据读取与独热编码
# -----------------------------
raw = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["sequence", "level"])
print(f"Loaded {len(raw):,} rows from {DATA_PATH}")

bases = ["A", "G", "C", "T"]
feature_names = [f"Position{p+1}_{b}" for p in range(9) for b in bases]

def one_hot_encode(seq: str) -> list[int]:
    return [1 if nt == b else 0 for nt in seq for b in bases]

X = pd.DataFrame(np.vstack(raw["sequence"].map(one_hot_encode)), columns=feature_names)
y = raw["level"].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)

# -----------------------------
# 2. 网格搜索 + 5 折交叉验证
# -----------------------------
param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
    "n_estimators": [100, 200, 300],
}

xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

grid = GridSearchCV(
    xgb,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

best_model: XGBRegressor = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# -----------------------------
# 3. 测试集 & 10 折 CV 评估
# -----------------------------
y_pred = best_model.predict(X_test)
print("\n=== Hold‑out Test Metrics ===")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(best_model, X, y, cv=kf, scoring="neg_root_mean_squared_error").mean()
cv_mae  = -cross_val_score(best_model, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
cv_r2   =  cross_val_score(best_model, X, y, cv=kf, scoring="r2").mean()
print("\n=== 10‑Fold CV Metrics (mean) ===")
print(f"RMSE: {cv_rmse:.4f}")
print(f"MAE : {cv_mae:.4f}")
print(f"R²  : {cv_r2:.4f}")

# -----------------------------
# 4. 特征重要性
# -----------------------------
importance = best_model.get_booster().get_score(importance_type="weight")
imp_df = (
    pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())})
    .sort_values(by="Importance", ascending=False)
)
plt.figure(figsize=(12, 6))
sns.barplot(x="Feature", y="Importance", data=imp_df)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("K‑mer Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. SHAP 解释
# -----------------------------
print("\nComputing SHAP values – this may take a while …")
explainer = shap.TreeExplainer(best_model)
shap_values_train = explainer.shap_values(X_train)
shap_values_test  = explainer.shap_values(X_test)
shap_inter_test   = explainer.shap_interaction_values(X_test)

shap.initjs()

# 5.1 Summary Plots
for tag, sv, data in [("train", shap_values_train, X_train), ("test", shap_values_test, X_test)]:
    shap.summary_plot(sv, data, plot_type="bar")
    shap.summary_plot(sv, data)

# 5.2 Force Plots（前三个测试样本）
for i in range(3):
    shap.force_plot(
        explainer.expected_value,
        shap_values_test[i],
        X_test.iloc[i],
        matplotlib=True,
    )
    plt.show()

# 5.3 Waterfall Plot（第一个测试样本）
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_test[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns,
    )
)

# 5.4 Interaction Plots
interaction_pairs = [
    ("Position7_T", "Position6_T"),
    ("Position7_G", "Position5_A"),
    ("Position2_G", "Position7_T"),
]
for p in interaction_pairs:
    shap.dependence_plot(
        p,
        shap_inter_test,
        X_test,
        interaction_index=None,
    )

# -----------------------------
# 6. 残差散点图
# -----------------------------
residuals = y_test - y_pred
plt.figure(figsize=(6, 5))
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Residual Plot")
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()
