import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ---------------------- 核心配置（数据路径为 train.csv）----------------------
DATA_PATH = "train.csv"  # 数据文件：train.csv
MODEL_DIR = "models"  # 模型保存目录
RANDOM_STATE = 42  # 随机种子
TEST_SIZE = 0.2  # 测试集比例

# XGBoost 原生参数（最基础配置，兼容所有版本）
XGB_PARAMS = {
    'eta': 0.08,  # 学习率
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'seed': RANDOM_STATE,
    'silent': 1  # 静默模式
}
NUM_BOOST_ROUNDS = 200  # 固定迭代次数（去掉早停，避免兼容问题）


# ---------------------- 数据加载与预处理 ----------------------
def load_and_preprocess_data(data_path):
    print(f"📊 加载数据：{data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 未找到数据文件：{data_path}")

    df = pd.read_csv(data_path)

    # 必要列检查
    required_cols = ['distance_traveled', 'num_of_passengers', 'fare']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ 数据缺少列：{missing_cols}")

    # 数据清洗
    df = df.dropna(subset=required_cols)
    df = df[(df['distance_traveled'] >= 0.1) & (df['distance_traveled'] <= 100.0)]
    df = df[(df['num_of_passengers'] >= 1) & (df['num_of_passengers'] <= 6)]
    df = df[(df['fare'] >= 0) & (df['fare'] <= 500.0)]

    # 特征工程
    df['distance_sq'] = df['distance_traveled'] ** 2
    df['passenger_distance'] = df['num_of_passengers'] * df['distance_traveled']
    df['is_high_passenger'] = (df['num_of_passengers'] >= 3).astype(int)

    print(f"✅ 数据预处理完成，有效样本数：{len(df)}")
    return df


# ---------------------- 模型训练（最基础写法，无任何兼容问题）----------------------
def train_model(df):
    feature_cols = [
        'distance_traveled',
        'num_of_passengers',
        'distance_sq',
        'passenger_distance',
        'is_high_passenger'
    ]
    X = df[feature_cols]
    y = df['fare']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"📈 训练集：{len(X_train)} 样本，测试集：{len(X_test)} 样本")
    # 划分训练集/测试集后，添加：
    X_test.to_csv("models/X_test.csv", index=False)
    y_test.to_csv("models/y_test.csv", index=False)
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为 XGBoost 原生 DMatrix 格式
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    # 训练模型（去掉早停、去掉所有复杂参数，仅保留核心）
    print("🚀 训练 XGBoost 模型...")
    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        verbose_eval=10  # 每10轮打印一次日志
    )

    # 预测（去掉 ntree_limit，直接预测）
    y_pred = model.predict(dtest)

    # 评估
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n📊 模型性能：")
    print(f"R² 分数：{r2:.4f}")
    print(f"RMSE：{rmse:.2f} 美元")

    return model, scaler, r2, rmse, feature_cols


# ---------------------- 模型保存 ----------------------
def save_model(model, scaler, r2, rmse, feature_cols):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 保存模型（原生格式）、标准化器、指标
    model.save_model(os.path.join(MODEL_DIR, "best_model_XGBoost.model"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "model_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"模型：XGBoost（兼容低版本）\n")
        f.write(f"训练时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"R²：{r2:.4f}\n")
        f.write(f"RMSE：{rmse:.2f} 美元\n")
        f.write(f"迭代轮数：{NUM_BOOST_ROUNDS}\n")
        f.write(f"特征列：{feature_cols}")

    print(f"\n💾 模型保存至：{MODEL_DIR}")


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    try:
        print("=" * 50)
        print("🚕 纽约出租车车费预测 - 模型训练")
        print("=" * 50)

        df = load_and_preprocess_data(DATA_PATH)
        model, scaler, r2, rmse, feature_cols = train_model(df)
        save_model(model, scaler, r2, rmse, feature_cols)

        print("\n🎉 训练完成！")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 训练失败：{str(e)}")
        raise