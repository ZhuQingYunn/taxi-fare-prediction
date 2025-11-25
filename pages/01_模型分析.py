# pages/01_模型分析.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

# 加载模型、标准化器、测试集数据（需提前保存测试集）
@st.cache_resource
def load_data_and_model():
    model = xgb.Booster()
    model.load_model("models/best_model_XGBoost.model")
    scaler = joblib.load("models/scaler.pkl")
    # 需提前在train.py中保存测试集：X_test.to_csv("models/X_test.csv", index=False)
    X_test = pd.read_csv("models/X_test.csv")
    y_test = pd.read_csv("models/y_test.csv")  # 同理保存y_test
    return model, scaler, X_test, y_test

model, scaler, X_test, y_test = load_data_and_model()

# 计算预测值
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(xgb.DMatrix(X_test_scaled))
residuals = y_test - y_pred  # 残差

st.title("📊 模型性能分析")

# 1. 残差图
st.subheader("1. 残差图（预测值 vs 残差）")
fig1, ax1 = plt.subplots(figsize=(8,4))
sns.scatterplot(x=y_pred, y=residuals, ax=ax1, alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel("预测车费")
ax1.set_ylabel("残差（实际-预测）")
st.pyplot(fig1)

# 2. 预测值vs实际值图
st.subheader("2. 预测值 vs 实际值")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.scatterplot(x=y_test, y=y_pred, ax=ax2, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x对角线
ax2.set_xlabel("实际车费")
ax2.set_ylabel("预测车费")
st.pyplot(fig2)

# 3. 学习曲线（需在train.py中保存训练过程的性能数据）
st.subheader("3. 学习曲线（训练/验证集RMSE）")
# 需在train.py训练时保存学习曲线数据：
# 例：把watchlist的日志存到log.txt，再读取
if "models/train_log.txt" in st.secrets:
    log_data = pd.read_csv("models/train_log.txt")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.lineplot(x=log_data["round"], y=log_data["train_rmse"], ax=ax3, label="训练集")
    sns.lineplot(x=log_data["round"], y=log_data["eval_rmse"], ax=ax3, label="验证集")
    ax3.set_xlabel("迭代轮数")
    ax3.set_ylabel("RMSE")
    ax3.legend()
    st.pyplot(fig3)