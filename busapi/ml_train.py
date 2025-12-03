#bus_model.pkl : 예측 모델 의미
# ml_train.py

import os
from pathlib import Path
from math import sqrt
import joblib
import pandas as pd
from django.conf import settings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from .models import bus_arrival_past


def load_from_db() -> pd.DataFrame:
    qs = bus_arrival_past.objects.values(
        "routeid", "timestamp", "remainseatcnt1", "station_num"
    )
    df = pd.DataFrame.from_records(qs)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["remainseatcnt1"] = pd.to_numeric(df["remainseatcnt1"], errors="coerce")
    df["station_num"] = pd.to_numeric(df["station_num"], errors="coerce")
    df["routeid"] = df["routeid"].astype(str)

    df = df.dropna(subset=["timestamp", "remainseatcnt1", "station_num", "routeid"])

    df["time_min"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    return df


def add_time_slots(df: pd.DataFrame) -> pd.DataFrame:
    start_min = 5 * 60 + 45  # 5:45
    end_min = 9 * 60 + 15

    df = df[(df["time_min"] >= start_min) & (df["time_min"] < end_min)].copy()

    df["slot_center_min"] = (
        ((df["time_min"] - start_min) // 30) * 30 + start_min + 15
    )

    return df


def build_slot_level_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["routeid", "station_num", "slot_center_min"], as_index=False)
          .agg(y=("remainseatcnt1", "mean"))
    )
    return agg


def train_model_and_save(model_path="bus_model.pkl") -> float:
    df = load_from_db()
    df = add_time_slots(df)
    agg = build_slot_level_table(df)

    # 🔥 routeid One-hot
    routeid_dummies = pd.get_dummies(agg["routeid"], prefix="routeid")

    df_train = pd.concat([agg, routeid_dummies], axis=1)

    feature_cols = ["station_num", "slot_center_min"] + list(routeid_dummies.columns)

    X = df_train[feature_cols]
    y = df_train["y"]

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
    rmse = sqrt(mean_squared_error(y, y_pred))
    print(f"[train RMSE] {rmse:.3f}")

    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "routeid_columns": list(routeid_dummies.columns),
    }

    model_abspath = Path(settings.BASE_DIR) / "busapi" / model_path
    joblib.dump(payload, model_abspath)

    return rmse
