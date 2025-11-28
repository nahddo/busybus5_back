from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import pandas as pd
from .models import bus_arrival_past
from .ml_train import train_model_and_save
from .ml_predict import predict_remaining_seats

def train_from_db():
    # db(bus_arrival_past)에서 훈련 데이터를 가져와 학습함수에 전달
    qs = bus_arrival_past.objects.all()
    df = pd.DataFrame(list(qs.values()))
    #train_model_and_save(df)

def run_training(request):
    train_from_db()
    return JsonResponse({"ok": True, "message": "Model training completed"})

def predict_seat(request):
    # 1) 프론트에서 보내줄 쿼리 파라미터 받기
    routeid = request.GET.get('routeid')
    select_time = request.GET.get('select_time')

    # 2) 간단한 유효성 검사 (없으면 에러 반환)
    if not routeid or not select_time:
        return JsonResponse(
            {"ok": False, "error": "route, stationId, time 파라미터가 필요합니다."},
            status=400,
        )
    try:
        routeid_int = int(routeid)
        select_time_int = int(select_time)
    except ValueError:
        return JsonResponse(
            {"ok": False, "error": "routeId와 select_time은 정수여야 합니다."},
            status=400,
        )

    # 3) 예측
    try:
        predications = predict_remaining_seats(routeid_int, routeid_int)
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": f"prediction error: {e}"},
            status=500,
        )


    data = {
        "routeid": routeid,
        "select_time": select_time,
        "predications": predications, #list 형태

    }
    return JsonResponse(data, status=200)