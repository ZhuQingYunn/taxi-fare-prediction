# pages/01_模型分析.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 页面配置
st.title("📊 模型性能深度分析")
st.subheader("基于 XGBoost 的车费预测模型评估")
st.divider()

# 全局绘图样式
plt.rcParams['font.sans-serif'] = ['Arial']  # 避免中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sns.set_style("whitegrid")


# ---------------------- 加载数据与模型（修复 y_test 格式）----------------------
@st.cache_resource
def load_data_and_model():
    # 模型路径
    model_path = "models/best_model_XGBoost.model"
    scaler_path = "models/scaler.pkl"
    X_test_path = "models/X_test.csv"
    y_test_path = "models/y_test.csv"

    # 检查文件完整性
    required_files = [model_path, scaler_path, X_test_path, y_test_path]
    missing_files = [f for f in required_files if not pd.io.common.file_exists(f)]
    if missing_files:
        st.error(f"❌ 缺少必要文件：{', '.join(missing_files)}")
        st.info("提示：请重新运行 train.py 训练模型（确保保存 X_test.csv 和 y_test.csv）")
        st.stop()

    # 加载模型和标准化器
    model = xgb.Booster()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # 加载测试集（关键修改：用 squeeze() 将 y_test 转为一维数组）
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()  # 单列DataFrame → 一维Series/数组

    return model, scaler, X_test, y_test


# 执行加载
model, scaler, X_test, y_test = load_data_and_model()

# ---------------------- 计算核心指标与预测值 ----------------------
# 标准化测试集特征
X_test_scaled = scaler.transform(X_test)
# 转换为 DMatrix 格式（适配低版本 XGBoost）
dtest = xgb.DMatrix(X_test_scaled)
# 预测
y_pred = model.predict(dtest)
# 计算残差（实际值 - 预测值）
residuals = y_test - y_pred
# 计算核心性能指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# ---------------------- 显示核心性能指标 ----------------------
st.subheader("🎯 核心性能指标")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² 决定系数", f"{r2:.4f}", help="越接近1，模型拟合效果越好")
with col2:
    st.metric("RMSE 均方根误差", f"${rmse:.2f}", help="越小，预测误差越小")
with col3:
    st.metric("MAE 平均绝对误差", f"${mae:.2f}", help="越小，平均预测偏差越小")
st.divider()

# ---------------------- 1. 残差图（预测值 vs 残差）----------------------
st.subheader("1. 残差分布分析")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 残差散点图（检查随机性）
sns.scatterplot(x=y_pred, y=residuals, ax=ax1, alpha=0.5, color="#2196F3")
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel("预测车费（美元）")
ax1.set_ylabel("残差（实际 - 预测）")
ax1.set_title("残差 vs 预测值（随机分布=良好）")
ax1.grid(alpha=0.3)

# 残差直方图（检查正态分布）
sns.histplot(residuals, ax=ax2, kde=True, color="#4CAF50", bins=30)
ax2.set_xlabel("残差（美元）")
ax2.set_ylabel("频次")
ax2.set_title("残差分布（接近正态=良好）")
ax2.grid(alpha=0.3)

st.pyplot(fig1)
st.caption("说明：残差应随机分布在0线附近，且直方图接近正态分布，表明模型无系统性误差")
st.divider()

# ---------------------- 2. 预测值 vs 实际值图 ----------------------
st.subheader("2. 预测值 vs 实际值")
fig2, ax = plt.subplots(figsize=(8, 6))

# 散点图
sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.5, color="#FF9800")
# 理想拟合线（y=x）
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', linewidth=2, label="理想拟合线（y=x）")

ax.set_xlabel("实际车费（美元）")
ax.set_ylabel("预测车费（美元）")
ax.set_title("预测值 vs 实际值（越贴近红线越好）")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig2)
st.caption("说明：点越贴近红色对角线，表明预测值与实际值越一致")
st.divider()

# ---------------------- 3. 误差分布箱线图 ----------------------
st.subheader("3. 误差分段分析")
# 按距离分段分析误差
X_test_with_pred = X_test.copy()
X_test_with_pred['实际车费'] = y_test
X_test_with_pred['预测车费'] = y_pred
X_test_with_pred['绝对误差'] = np.abs(residuals)

# 距离分段
X_test_with_pred['距离分段'] = pd.cut(
    X_test_with_pred['distance_traveled'],
    bins=[0, 3, 10, 100],
    labels=['短途（<3km）', '中途（3-10km）', '长途（>10km）']
)

fig3, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(
    x='距离分段',
    y='绝对误差',
    data=X_test_with_pred,
    ax=ax,
    palette=["#9C27B0", "#2196F3", "#FF5722"]
)
ax.set_xlabel("行程距离分段")
ax.set_ylabel("绝对误差（美元）")
ax.set_title("不同距离分段的预测误差")
ax.grid(alpha=0.3, axis='y')

st.pyplot(fig3)
st.caption("说明：箱线图展示各距离段误差的分布，可观察模型在不同场景下的预测稳定性")
st.divider()

# ---------------------- 4. 特征重要性分析 ----------------------
st.subheader("4. 模型特征重要性")
# 提取XGBoost特征重要性
feature_names = [
    '行驶距离', '乘客数量', '距离平方', '乘客×距离', '是否多人出行'
]
feature_importance = model.get_score(importance_type='gain')  # 按增益计算重要性

# 匹配特征名与重要性
importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': [feature_importance.get(f'f{i}', 0) for i in range(len(feature_names))]
}).sort_values('重要性', ascending=True)

fig4, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x='重要性',
    y='特征',
    data=importance_df,
    ax=ax,
    color="#FFC107"
)
ax.set_xlabel("特征重要性（增益值）")
ax.set_ylabel("特征名称")
ax.set_title("XGBoost 特征重要性排序")
ax.grid(alpha=0.3, axis='x')

st.pyplot(fig4)
st.caption("说明：重要性越高，该特征对车费预测的影响越大")