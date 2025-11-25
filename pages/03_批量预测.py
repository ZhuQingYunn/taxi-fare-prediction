import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🗺️ 纽约出租车行程分析")
st.subheader("距离-车费关系分布")
st.divider()

# 加载数据（无需经纬度）
try:
    df = pd.read_csv("train.csv").sample(10000)
    # 距离分段
    df['distance_segment'] = pd.cut(
        df['distance_traveled'],
        bins=[0, 3, 10, 100],
        labels=['短途（<3km）', '中途（3-10km）', '长途（>10km）']
    )
except Exception as e:
    st.error(f"❌ 数据加载失败：{str(e)}")
    st.info("提示：请确保train.csv包含distance_traveled、fare列")
    st.stop()

# 交互控件
distance_segment = st.selectbox(
    "选择行程距离分段",
    options=df['distance_segment'].unique(),
    index=0
)
df_filtered = df[df['distance_segment'] == distance_segment]

# 距离-车费散点图
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(
    df_filtered['distance_traveled'],
    df_filtered['fare'],
    alpha=0.3,
    c=df_filtered['fare'],
    cmap='viridis'
)
ax.set_xlabel("行驶距离（km）")
ax.set_ylabel("车费（美元）")
ax.set_title(f"{distance_segment} 距离-车费分布")
ax.grid(alpha=0.3)
plt.colorbar(ax.collections[0], label='车费（美元）')
st.pyplot(fig)

# 统计信息
st.subheader("📊 统计摘要")
st.write(f"- 总行程数：{len(df_filtered)} 条")
st.write(f"- 平均距离：{df_filtered['distance_traveled'].mean():.2f} km")
st.write(f"- 平均车费：${df_filtered['fare'].mean():.2f}")