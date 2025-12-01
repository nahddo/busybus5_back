from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from .models import bus_arrival_past
from django.contrib.auth.decorators import user_passes_test

try:
    from .ml_train import train_model_and_save
except ImportError:
    train_model_and_save = None

try:
    from .ml_predict import predict_remaining_seats
except ImportError:
    def predict_remaining_seats(routeid_int, select_time_int):
        return []

@user_passes_test(lambda u: u.is_superuser)
def run_training(request):
    try:
        rmse = train_model_and_save()  # 모델 학습
        return JsonResponse({"ok": True, "rmse": rmse})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)

def predict_seat(request):
    routeid = request.GET.get("routeid")
    select_time = request.GET.get("select_time")

    if not routeid or not select_time:
        return JsonResponse(
            {"error": "routeid, select_time 파라미터가 필요합니다."},
            status=400,
        )

    try:
        routeid_int = int(routeid)
        select_time_int = int(select_time)
    except ValueError:
        return JsonResponse(
            {"error": "routeid와 select_time은 정수여야 합니다."},
            status=400,
        )

    try:
        predictions = predict_remaining_seats(routeid_int, select_time_int)
    except Exception as e:
        import traceback
        print('error during prediction')
        print(traceback.format_exc())
        return JsonResponse(
            {"error": f"prediction error: {e}"},
            status=500,
        )

    # 프론트엔드 타입 정의와 일치하도록 응답 형식 조정
    # PredictSeatResponse: { routeid, select_time, predictions, error? }
    return JsonResponse(
        {
            "routeid": routeid_int,
            "select_time": select_time_int,
            "predictions": predictions,
        },
        status=200,
    )



@csrf_exempt
@require_GET
def bus_realtime(request):
    """
    버스 노선별 실시간 버스 위치 조회
    - 쿼리 파라미터: routeid, service_date, time_slot
    - 반환: 해당 노선의 현재 버스 위치 정보 (vehid1, stationid, station_num 등)
    - 잔여좌석 정보는 예측 모델에서 따로 받으므로 여기서는 제외
    - routeid에 따라 현재 버스의 위치를 실시간 API에서 받아서 반환
    """
    routeid = request.GET.get('routeid')
    service_date = request.GET.get('service_date')
    time_slot = request.GET.get('time_slot')

    if not routeid or not service_date or not time_slot:
        return JsonResponse(
            {"error": "routeid, service_date, time_slot 파라미터가 필요합니다."},
            status=400,
        )

    try:
        # routeid를 정수로 변환
        try:
            routeid_int = int(routeid)
        except ValueError:
            return JsonResponse(
                {"error": "routeid는 정수여야 합니다."},
                status=400,
            )

        # routeid에 따라 실시간 버스 위치 데이터 조회
        # 현재 버스의 위치를 실시간 API에서 받아서 반환
        # TODO: 실제 실시간 버스 위치 API를 호출하여 현재 버스 위치를 가져와야 함
        # 현재는 과거 데이터에서 최신 데이터를 조회하는 방식 (실시간 API 연동 필요)
        from django.db.models import Max

        # routeid로 필터링하여 최근 데이터 조회
        # 각 station_num별로 가장 최근 timestamp를 가진 데이터만 가져오기
        # (실제 실시간 API에서는 현재 버스 위치를 직접 받아옴)
        latest_records = (
            bus_arrival_past.objects
            .filter(routeid=routeid_int)
            .values('station_num')
            .annotate(latest_timestamp=Max('timestamp'))
        )

        # 각 정류장의 최신 데이터 조회 (실시간 API 연동 시 이 부분 수정 필요)
        data = []
        for record in latest_records:
            station_num = record['station_num']
            latest_timestamp = record['latest_timestamp']

            # 해당 정류장의 최신 데이터 가져오기
            latest_data = (
                bus_arrival_past.objects
                .filter(
                    routeid=routeid_int,
                    station_num=station_num,
                    timestamp=latest_timestamp
                )
                .first()
            )

            if latest_data:
                # 실시간 API에서 받은 현재 버스 위치 정보만 반환
                # vehid1: 버스 ID (현재 버스 위치 표시용)
                # stationid: 정류장 ID (정류장 매핑용)
                # 잔여좌석 정보는 예측 모델에서 따로 받으므로 제외
                data.append({
                    "service_date": service_date,
                    "arrival_time": service_date,  # 실제 arrival_time이 없으므로 service_date 사용
                    "vehid1": str(latest_data.vehid1),  # 버스 ID (현재 버스 위치 표시용)
                    "station_num": str(latest_data.station_num),  # 정류장 번호
                    "routeid": str(routeid_int),  # 노선 ID
                    "routename": str(routeid_int),  # routename은 별도 조회 필요 시 수정
                    "stationid": f"234{latest_data.station_num:06d}",  # stationid 생성 (실제 DB에 있으면 수정 필요)
                    "crowded_level": 1,  # 기본값 (실제 데이터가 있으면 수정 필요)
                })

        return JsonResponse(data, status=200, safe=False)
    except Exception as e:
        return JsonResponse(
            {"error": f"서버 오류: {str(e)}"},
            status=500,
        )


@csrf_exempt
@require_GET
def station_realtime(request):
    """
    정류장별 실시간 데이터 조회
    - 쿼리 파라미터: stationid, service_date, time_slot
    - 반환: 해당 정류장을 지나가는 모든 버스의 실시간 데이터 배열
    """
    stationid = request.GET.get('stationid')
    service_date = request.GET.get('service_date')
    time_slot = request.GET.get('time_slot')

    if not stationid or not service_date or not time_slot:
        return JsonResponse(
            {"error": "stationid, service_date, time_slot 파라미터가 필요합니다."},
            status=400,
        )

    try:
        # TODO: 실제 실시간 데이터 소스에서 데이터를 가져와야 합니다.
        # 현재는 빈 배열을 반환합니다. 실제 데이터베이스나 외부 API와 연동 필요
        # 예시: bus_arrival_past 모델에서 해당 정류장의 최근 데이터를 조회하거나 외부 API 호출

        # 임시로 빈 배열 반환 (프론트엔드가 에러 없이 동작하도록)
        data = []

        return JsonResponse(data, status=200, safe=False)
    except Exception as e:
        return JsonResponse(
            {"error": f"서버 오류: {str(e)}"},
            status=500,
        )


@csrf_exempt
@require_GET
def recommend_route(request):
    """
    경로 추천 API
    - 쿼리 파라미터: origin_stationid, dest_stationid, weekday, time_slot, time_type, fast_option
    - 반환: 추천 경로 정보 (버스 번호, routeid, 소요 시간 등)

    파라미터 설명:
    - origin_stationid: 출발 정류장 ID
    - dest_stationid: 도착 정류장 ID
    - weekday: 요일 (월요일, 화요일, 수요일, 목요일, 금요일)
    - time_slot: 시간대 (6:00, 6:30, 7:00, 7:30, 8:00, 8:30, 9:00)
    - time_type: 시간 타입 (도착시간, 출발시간)
    - fast_option: 최적화 옵션 (최단시간, 최소대기)
    """
    origin_stationid = request.GET.get('origin_stationid')
    dest_stationid = request.GET.get('dest_stationid')
    weekday = request.GET.get('weekday', '월요일')
    time_slot = request.GET.get('time_slot', '8:30')
    time_type = request.GET.get('time_type', '도착시간')
    fast_option = request.GET.get('fast_option', '최단시간')

    # 필수 파라미터 검증
    if not origin_stationid or not dest_stationid:
        return JsonResponse(
            {"ok": False, "error": "origin_stationid와 dest_stationid 파라미터가 필요합니다."},
            status=400,
        )

    try:
        # TODO: 실제 경로 추천 알고리즘 구현
        # 현재는 기본 구조만 반환합니다.
        # 향후 구현 시 고려사항:
        # 1. 정류장 간 경로 탐색 (직행, 환승 등)
        # 2. 예측 좌석 수를 활용한 혼잡도 고려
        # 3. 시간대별 소요 시간 예측
        # 4. fast_option에 따른 최적 경로 선택

        # 시간 슬롯에서 시간만 추출 (예: "8:30" -> "8")
        select_time = time_slot.split(":")[0] if ":" in time_slot else time_slot

        # 기본 응답 구조 (프론트엔드가 에러 없이 동작하도록)
        data = {
            "ok": True,
            "origin_stationid": origin_stationid,
            "dest_stationid": dest_stationid,
            "weekday": weekday,
            "time_slot": time_slot,
            "time_type": time_type,
            "fast_option": fast_option,
            "recommended_route": {
                "bus_numbers": [],  # 추천 버스 번호 목록
                "routeid": None,  # 추천 routeid
                "duration_minutes": None,  # 예상 소요 시간 (분)
                "congestion_level": None,  # 예상 혼잡도
            },
            "message": "경로 추천 기능은 현재 개발 중입니다."
        }

        return JsonResponse(data, status=200)
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": f"서버 오류: {str(e)}"},
            status=500,
        )