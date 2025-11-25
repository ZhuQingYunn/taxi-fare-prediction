import streamlit as st
import folium
import pandas as pd
import numpy as np
from streamlit_folium import st_folium  # 需安装：pip install streamlit-folium
import joblib

# 页面配置
st.title("🗺️ 纽约出租车行程地理分布")
st.subheader("高频率行程热点图 + 距离-车费关系")
st.divider()


# 加载数据和模型（用于关联车费预测）
@st.cache_resource
def load_data_and_model():
    # 加载训练数据（需包含经纬度列，若数据量大可抽样）
    df = pd.read_csv("train.csv").sample(10000)  # 抽样1万条数据，避免加载过慢
    scaler = joblib.load("models/scaler.pkl")

    # 加载模型（用于实时计算车费）
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model("models/best_model_XGBoost.model")
    return df, model, scaler


# 加载数据（处理可能的经纬度缺失）
try:
    df, model, scaler = load_data_and_model()

    # 过滤纽约市经纬度范围（避免异常值）
    nyc_bounds = {
        'lat_min': 40.5, 'lat_max': 41.0,
        'lon_min': -74.3, 'lon_max': -73.7
    }
    df = df[
        (df['pickup_lat'].between(nyc_bounds['lat_min'], nyc_bounds['lat_max'])) &
        (df['pickup_lon'].between(nyc_bounds['lon_min'], nyc_bounds['lon_max'])) &
        (df['dropoff_lat'].between(nyc_bounds['lat_min'], nyc_bounds['lat_max'])) &
        (df['dropoff_lon'].between(nyc_bounds['lon_min'], nyc_bounds['lon_max']))
        ].reset_index(drop=True)

    # 计算行程中点（用于绘制热点）
    df['mid_lat'] = (df['pickup_lat'] + df['dropoff_lat']) / 2
    df['mid_lon'] = (df['pickup_lon'] + df['dropoff_lon']) / 2

    # 按距离分段（用于热点颜色区分）
    df['distance_segment'] = pd.cut(
        df['distance_traveled'],
        bins=[0, 3, 10, 100],
        labels=['短途（<3km）', '中途（3-10km）', '长途（>10km）']
    )

except Exception as e:
    st.error(f"❌ 数据加载失败：{str(e)}")
    st.info("提示：请确保 train.csv 包含 pickup_lat、pickup_lon、dropoff_lat、dropoff_lon 列，且 models 文件夹已上传完整")
    st.stop()

# 交互控件：选择距离分段
distance_segment = st.selectbox(
    "选择行程距离分段",
    options=df['distance_segment'].unique(),
    index=0
)
df_filtered = df[df['distance_segment'] == distance_segment]

# 绘制纽约地图 + 热点图
st.subheader(f"{distance_segment} 行程热点分布（抽样{len(df_filtered)}条数据）")
m = folium.Map(
    location=[40.7128, -74.0060],  # 纽约市中心经纬度
    zoom_start=11,
    tiles="CartoDB positron"  # 简洁地图样式
)

# 添加热点标记（按车费颜色区分）
for idx, row in df_filtered.iterrows():
    # 车费颜色映射（红色=高车费，蓝色=低车费）
    fare_color = folium.ColorGradient(
        colors=['blue', 'green', 'orange', 'red'],
        vmin=df['fare'].min(),
        vmax=df['fare'].max()
    ).get_color(row['fare'])

    # 添加行程中点标记
    folium.CircleMarker(
        location=[row['mid_lat'], row['mid_lon']],
        radius=3,
        color=fare_color,
        fill=True,
        fill_color=fare_color,
        fill_opacity=0.6,
        popup=f"""
        距离：{row['distance_traveled']:.2f}km<br>
        乘客：{row['num_of_passengers']}人<br>
        车费：${row['fare']:.2f}
        """
    ).add_to(m)

# 添加上下车点连线（随机选100条，避免地图混乱）
sample_df = df_filtered.sample(min(100, len(df_filtered)))
for idx, row in sample_df.iterrows():
    folium.PolyLine(
        locations=[
            [row['pickup_lat'], row['pickup_lon']],
            [row['dropoff_lat'], row['dropoff_lon']]
        ],
        color='gray',
        weight=1,
        opacity=0.5
    ).add_to(m)

# 在Streamlit中显示地图
st_folium(m, width=800, height=500)

# 距离-车费关系散点图
st.divider()
st.subheader(f"{distance_segment} 距离-车费关系")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']  # 避免中文乱码

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
st.write(
    f"- 最高车费：${df_filtered['fare'].max():.2f}（距离：{df_filtered.loc[df_filtered['fare'].idxmax(), 'distance_traveled']:.2f}km）")