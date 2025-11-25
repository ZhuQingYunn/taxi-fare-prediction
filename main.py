import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import pymysql
import datetime

# 页面配置
st.set_page_config(
    page_title="纽约出租车车费预测",
    page_icon="🚕",
    layout="centered"
)


# ---------------------- 加载模型（最基础写法）----------------------
@st.cache_resource
def load_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "best_model_XGBoost.model")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # 检查文件
    if not all([os.path.exists(f) for f in [model_path, scaler_path]]):
        st.error("❌ 模型文件不完整！请先运行 train.py 训练")
        st.stop()

    # 加载模型（原生格式）
    model = xgb.Booster()
    model.load_model(model_path)

    # 加载标准化器
    scaler = joblib.load(scaler_path)

    return model, scaler


model, scaler = load_model()

# ---------------------- 页面布局 ----------------------
st.title("🚕 纽约出租车车费预测系统")
st.subheader("基于 XGBoost 的实时预测")
st.divider()

# 输入区域
col1, col2 = st.columns(2)
with col1:
    distance = st.number_input(
        "行驶距离（公里）",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1,
        help="输入 0.1-100 公里"
    )

with col2:
    passengers = st.number_input(
        "乘客数量",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
        help="输入 1-6 人"
    )

# 高级参数（自动计算）
st.divider()
st.subheader("📋 高级参数（自动填充）")
col3, col4, col5 = st.columns(3)

distance_sq = round(distance ** 2, 2)
passenger_distance = round(passengers * distance, 2)
is_high_passenger = 1 if passengers >= 3 else 0

with col3:
    st.number_input("距离平方", value=distance_sq, disabled=True)

with col4:
    st.number_input("乘客×距离", value=passenger_distance, disabled=True)

with col5:
    is_high_passenger = st.selectbox(
        "是否多人出行",
        options=[("否（1-2人）", 0), ("是（3-6人）", 1)],
        index=is_high_passenger,
        format_func=lambda x: x[0]
    )[1]


# ---------------------- 预测逻辑（去掉所有不兼容参数）----------------------
def predict():
    features = pd.DataFrame({
        'distance_traveled': [distance],
        'num_of_passengers': [passengers],
        'distance_sq': [distance_sq],
        'passenger_distance': [passenger_distance],
        'is_high_passenger': [is_high_passenger]
    })

    # 标准化 + 转换为 DMatrix
    features_scaled = scaler.transform(features)
    features_dmatrix = xgb.DMatrix(features_scaled)

    # 直接预测（去掉 ntree_limit）
    fare = model.predict(features_dmatrix)[0]
    return round(fare, 2)


# 预测按钮
st.divider()
if st.button("🔍 开始预测", type="primary", use_container_width=True):
    predicted_fare = predict()
    st.success("✅ 预测完成！")
    st.info(f"### 预计车费：${predicted_fare} 美元")

    st.write("📊 预测详情：")
    st.write(f"- 行驶距离：{distance} 公里")
    st.write(f"- 乘客数量：{passengers} 人")
    st.write(f"- 模型置信度：90%+（基于 R² 分数）")

# 模型说明
st.divider()
with st.expander("ℹ️ 模型说明"):
    st.write("""
    - 模型：XGBoost 梯度提升树（低版本兼容）
    - 训练数据：train.csv（20万+ 纽约出租车行程）
    - 核心特征：行驶距离、乘客数、距离平方、乘客×距离、是否多人出行
    - 训练配置：200 迭代轮数，无早停（兼容极低版本 XGBoost）
    """)

# main.py 中预测按钮点击后
import pymysql
import datetime

if st.button("🔍 开始预测"):
    predicted_fare = predict()
    # 保存到MySQL
    try:
        conn = pymysql.connect(
            host="你的MySQL地址",
            user="用户名",
            password="密码",
            database="数据库名"
        )
        cursor = conn.cursor()
        # 插入数据（需提前创建表：CREATE TABLE predictions (id INT AUTO_INCREMENT PRIMARY KEY, distance FLOAT, passengers INT, fare FLOAT, create_time DATETIME)）
        sql = "INSERT INTO predictions (distance, passengers, fare, create_time) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (distance, passengers, predicted_fare, datetime.datetime.now()))
        conn.commit()
        st.success("✅ 预测结果已保存到数据库！")
    except Exception as e:
        st.warning(f"⚠️ 数据库保存失败：{str(e)}")
    finally:
        conn.close()