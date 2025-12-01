# ml_predict.py

## 모델 사용하여 예측하는 코드

# 입력 : routeid, select_time 형태

# busapi/ml_predict.py

# slot_index(0~6)를 기반으로 예측하는 버전

from pathlib import Path

from typing import List, Dict

import joblib

import pandas as pd

from django.conf import settings

from .models import bus_arrival_past


def _load_model_payload(model_path: str = "bus_model.pkl"):
    """
    모델 로딩
    - 저장된 모델 파일을 로드하여 반환합니다.
    """
    model_abspath = Path(settings.BASE_DIR) / "busapi" / model_path
    payload = joblib.load(model_abspath)
    return payload


def _slot_index_to_center_min(slot_index: int) -> int:
    """
    slot_index (0~6) → slot_center_min 변환

    기준:
        0 → 06:00 → 360
        1 → 06:30 → 390
        ...
        6 → 09:00

    Args:
        slot_index: 시간 슬롯 인덱스 (0~6)

    Returns:
        slot_center_min: 분 단위로 변환된 시간 (예: 360 = 06:00)

    Raises:
        ValueError: slot_index가 0~6 범위를 벗어난 경우
    """
    if not 0 <= slot_index <= 6:
        raise ValueError("slot_index는 0~6이어야 합니다.")

    start_min = 5 * 60 + 45  # 345 = 05:45
    slot_center_min = start_min + (slot_index * 30) + 15
    return slot_center_min


def predict_remaining_seats(routeid: int, slot_index: int) -> List[Dict]:
    """
    routeid와 slot_index(0~7)를 기반으로
    해당 노선의 모든 station_num 잔여좌석 예측값 반환.

    Args:
        routeid: 노선 ID
        slot_index: 시간 슬롯 인덱스 (0~7)

    Returns:
        예측 결과 리스트 (각 정류장별 잔여좌석 예측값)
        [
            {
                "routeid": int,
                "station_num": int,
                "slot_index": int,
                "remainseat_pred": float
            },
            ...
        ]
    """
    payload = _load_model_payload()
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    # 1) 슬롯 index → slot_center_min 변환
    slot_center_min = _slot_index_to_center_min(slot_index)

    # 2) 해당 노선의 모든 정류장 번호 목록 가져오기
    station_nums = list(
        bus_arrival_past.objects.filter(routeid=routeid)
        .values_list("station_num", flat=True)
        .distinct()
        .order_by("station_num")
    )

    if not station_nums:
        return []

    # 3) 예측용 데이터프레임 생성
    rows = [
        {"routeid": routeid, "station_num": s, "slot_center_min": slot_center_min}
        for s in station_nums
    ]
    df_pred = pd.DataFrame(rows)
    X = df_pred[feature_cols]

    # 4) 모델 예측
    df_pred["y_pred"] = model.predict(X)

    # 5) 반환 형태
    results: List[Dict] = []
    for _, row in df_pred.iterrows():
        results.append(
            {
                "routeid": int(row["routeid"]),
                "station_num": int(row["station_num"]),
                "slot_index": slot_index,
                "remainseat_pred": float(row["y_pred"]),
            }
        )

    return results