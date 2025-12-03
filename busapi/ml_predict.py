# ml_predict.py

from pathlib import Path
import joblib
import pandas as pd
from typing import List, Dict

from django.conf import settings
from .models import bus_arrival_past


def _load_model_payload(model_path="bus_model.pkl"):
    path = Path(settings.BASE_DIR) / "busapi" / model_path
    return joblib.load(path)


def _slot_index_to_center_min(slot_index: int) -> int:
    return 345 + slot_index * 30 + 15  # 5:45 + slot*30 + 15


def predict_remaining_seats(routeid: str, slot_index: int) -> List[Dict]:
    payload = _load_model_payload()
    model = payload["model"]
    routeid_columns = payload["routeid_columns"]
    feature_cols = payload["feature_cols"]

    slot_center_min = _slot_index_to_center_min(slot_index)

    station_nums = (
        bus_arrival_past.objects.filter(routeid=str(routeid))
        .values_list("station_num", flat=True)
        .distinct()
    )
    station_nums = sorted(int(s) for s in station_nums)

    rows = []
    for s in station_nums:
        row = {
            "station_num": s,
            "slot_center_min": slot_center_min,
        }
        # one-hot routeid
        for col in routeid_columns:
            row[col] = 1 if col == f"routeid_{routeid}" else 0

        rows.append(row)

    df_pred = pd.DataFrame(rows)
    y_pred = model.predict(df_pred[feature_cols])

    results = []
    for s, pred in zip(station_nums, y_pred):
        pred = round(pred)
        pred = max(0, min(pred, 45))   # 0~45 제한

        results.append({
            "routeid": routeid,
            "station_num": s,
            "slot_index": slot_index,
            "remainseat_pred": pred,
        })

    return results