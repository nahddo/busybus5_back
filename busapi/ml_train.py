##모델 학습하는 코드
##bus_model.pkl : 예측 모델 의미
# ml_train.py

# busapi/ml_train.py

import os
from pathlib import Path
from math import sqrt
import joblib
import pandas as pd
from django.conf import settings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from .models import bus_arrival_past

# ------------------------------------
# 1. DB에서 로딩 & 기본 전처리
# ------------------------------------
def load_from_db() -> pd.DataFrame:
    qs = bus_arrival_past.objects.values(
        "routeid",
        "timestamp",
        "remainseatcnt1",
        "station_num",
    )
    df = pd.DataFrame.from_records(qs)
    if df.empty:
        raise ValueError("bus_arrival_past 테이블에 데이터가 없습니다.")

    # 타입 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["remainseatcnt1"] = pd.to_numeric(df["remainseatcnt1"], errors="coerce")
    df["station_num"] = pd.to_numeric(df["station_num"], errors="coerce")
    df["routeid"] = pd.to_numeric(df["routeid"], errors="coerce")

    # 이상치/결측 제거
    df = df.dropna(subset=["remainseatcnt1", "station_num", "routeid"])

    # 날짜/시간 파생
    df["service_date"] = df["timestamp"].dt.date
    df["time_min"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    return df


# ------------------------------------
# 2. 30분 slot 생성 (5:45~9:15 범위)
# ------------------------------------
def add_time_slots(df: pd.DataFrame) -> pd.DataFrame:
    start_min = 5 * 60 + 45  # 5:45 -> 345
    end_min = 9 * 60 + 15    # 9:15 -> 555

    df = df[(df["time_min"] >= start_min) & (df["time_min"] < end_min)].copy()
    if df.empty:
        raise ValueError("5:45~9:15 범위에 해당하는 데이터가 없습니다.")

    # [5:45~6:15) -> 6:00, [6:15~6:45) -> 6:30, ...
    df["slot_center_min"] = (
        ((df["time_min"] - start_min) // 30) * 30 + start_min + 15
    )

    return df


# ------------------------------------
# 3. routeid / station_num / slot별 평균 잔여좌석 집계
# ------------------------------------
def build_slot_level_table(df: pd.DataFrame) -> pd.DataFrame:
    # 여러 날의 데이터를 한꺼번에 평균
    agg = (
        df.groupby(["routeid", "station_num", "slot_center_min"], as_index=False)
          .agg(
              y=("remainseatcnt1", "mean"),
              n=("remainseatcnt1", "size"),
          )
    )
    if agg.empty:
        raise ValueError("집계 결과가 비어 있습니다.")
    return agg


# ------------------------------------
# 4. 모델 학습 + 저장
# ------------------------------------
def train_model_and_save(
    model_path: str = "bus_model.pkl",
) -> float:
    """
    DB에서 데이터를 읽어 XGBoost 회귀 모델 학습 후
    busapi/bus_model.pkl 로 저장.
    return 값: train RMSE
    """
    df = load_from_db()
    df = add_time_slots(df)
    agg = build_slot_level_table(df)

    feature_cols = ["routeid", "station_num", "slot_center_min"]
    X = agg[feature_cols]
    y = agg["y"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = sqrt(mse)
    print(f"[train RMSE] {rmse:.3f}")

    payload = {
        "model": model,
        "feature_cols": feature_cols,
    }

    model_abspath = Path(settings.BASE_DIR) / "busapi" / model_path
    joblib.dump(payload, model_abspath)
    print(f"model saved to {model_abspath}")

    return rmse