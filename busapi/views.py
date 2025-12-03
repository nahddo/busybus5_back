from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import user_passes_test
import pandas as pd
from .models import bus_arrival_past
import requests
import json
from django.conf import settings
from pathlib import Path


# 공공데이터포털 서비스 키
# SERVICE_KEY = "52f50a9dca9673918e8d195dab87644394bf9c85a814c758daedb44634df54c6"
SERVICE_KEY = "1cfef036ae8826960c98fdb06e237c675fcbbc27a26106b8865eec77ed9f1cf8"


USE_FAKE_REALTIME = False  # 🔥 개발용 플래그 (실제 운영 시 False 로 바꾸거나 이 블록 삭제)


# ... 기존 import들 위/아래 아무 데나 괜찮지만, 함수 정의보다 위에
DATA_DIR = Path(settings.BASE_DIR) / "busapi" / "data"

# 1) 정류장 → {name, busNums, busCount}
with open(DATA_DIR / "stationBus.json", encoding="utf-8") as f:
    STATION_BUS = json.load(f)

# 2) routeId → [ {route_nm, sta_order, station_id, station_nm}, ... ]
with open(DATA_DIR / "routes.json", encoding="utf-8") as f:
    ROUTES = json.load(f)

# 3) 버스번호(route_nm) → routeId 리스트 (대부분 1개일 가능성이 큼)
ROUTE_NM_TO_IDS: dict[str, list[str]] = {}
for route_id, stops in ROUTES.items():
    if not stops:
        continue
    route_nm = stops[0].get("route_nm")
    if not route_nm:
        continue
    ROUTE_NM_TO_IDS.setdefault(route_nm, []).append(route_id)


def get_local_route_stops(routeid: str):
    """local routes.json 에서 해당 노선의 정류장 목록을 가져온다."""
    return ROUTES.get(str(routeid), [])


def get_local_routes_via_station(stationid: str):
    """
    local stationBus.json + routes.json 으로
    '이 정류장을 지나는 노선들의 (routeId, routeName, staOrder)' 리스트를 만든다.
    """
    stationid = str(stationid)

    station_info = STATION_BUS.get(stationid)
    if not station_info:
        return []

    bus_nums = station_info.get("busNums", [])
    results = []

    for bus_nm in bus_nums:
        route_ids = ROUTE_NM_TO_IDS.get(bus_nm, [])
        for rid in route_ids:
            stops = ROUTES.get(rid, [])
            for stop in stops:
                if str(stop.get("station_id")) == stationid:
                    results.append(
                        {
                            "routeId": rid,
                            "routeName": bus_nm,
                            "staOrder": stop.get("sta_order"),
                        }
                    )
                    break  # 이 routeId 에서는 해당 station은 한 번만 나오면 됨

    return results


# 🔥 BusSearch / StationSearch 에서 쓸 "가짜 노선 실시간 데이터"
#   → 네가 위에 붙여준 긴 JSON 중 일부만 써도 되고, 통째로 써도 됨
FAKE_ROUTE_234001736 = [
    {
        "service_date": "2025-12-03",
        "arrival_time": "2025-12-03 19:24:45.979",
        "vehid1": "",
        "station_num": "1",
        "remainseat_at_arrival": None,
        "routeid": "234001736",
        "routename": "3302",
        "stationid": "234001276",
        "crowded_level": 2,
    },
    {
        "service_date": "2025-12-03",
        "arrival_time": "2025-12-03 19:24:50.214",
        "vehid1": "230010044",  # 🔥 여기부터 실제 버스
        "station_num": "3",
        "remainseat_at_arrival": 44,
        "routeid": "234001736",
        "routename": "3302",
        "stationid": "234001270",
        "crowded_level": 1,
    },
    {
        "service_date": "2025-12-03",
        "arrival_time": "2025-12-03 19:26:07.599",
        "vehid1": "230010044",
        "station_num": "8",
        "remainseat_at_arrival": 44,
        "routeid": "234001736",
        "routename": "3302",
        "stationid": "234000384",
        "crowded_level": 1,
    },
    {
        "service_date": "2025-12-03",
        "arrival_time": "2025-12-03 19:26:22.693",
        "vehid1": "230010045",
        "station_num": "31",
        "remainseat_at_arrival": 37,
        "routeid": "234001736",
        "routename": "3302",
        "stationid": "123000008",
        "crowded_level": 1,
    },
    {
        "service_date": "2025-12-03",
        "arrival_time": "2025-12-03 19:26:38.958",
        "vehid1": "230010042",
        "station_num": "45",
        "remainseat_at_arrival": 21,
        "routeid": "234001736",
        "routename": "3302",
        "stationid": "234001200",
        "crowded_level": 1,
    },
    # ... 필요하면 네가 붙여준 JSON에서 더 복사해서 추가
]



# -----------------------------
#  ML 관련 (그대로 유지)
# -----------------------------
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
        routeid_str = routeid
        select_time_int = int(select_time)
    except ValueError:
        return JsonResponse(
            {"error": "routeid와 select_time은 정수여야 합니다."},
            status=400,
        )

    try:
        predictions = predict_remaining_seats(routeid_str, select_time_int)
    except Exception as e:
        import traceback
        print("error during prediction")
        print(traceback.format_exc())
        return JsonResponse(
            {"error": f"prediction error: {e}"},
            status=500,
        )

    return JsonResponse(
        {
            "routeid": routeid_str,
            "select_time": select_time_int,
            "predictions": predictions,
        },
        status=200,
    )


# -----------------------------
#  bus_realtime  (버스 번호 화면용)
# -----------------------------

@csrf_exempt
def bus_realtime(request):
    """
    GET /api/bus/realtime/?routeid=234001736
        → 노선 단위 실시간 (BusRouteCard 에서 사용 가능)

    POST /api/bus/realtime/
        body: {
          "routeId": "234001736",
          "stations": [
            { "stationId": "234001276", "staOrder": 1 },
            ...
          ]
        }
        → BusSearch 에서 사용하는 형식 유지
           결과 형식:
        {
          "routeId": "...",
          "results": [
            { "stationId": "...", "staOrder": 1, "raw": { ... } },
            ...
          ]
        }
    """

    # --------------------
    # 공통: 노선 실시간 위치 API (한 번만 호출)
    # --------------------
    def call_buslocation_api(routeid: str):
        URL_LOC = "https://apis.data.go.kr/6410000/buslocationservice/v2/getBusLocationListv2"
        try:
            r = requests.get(
                URL_LOC,
                params={
                    "serviceKey": SERVICE_KEY,
                    "routeId": routeid,
                    "format": "json",
                },
                timeout=5,
            )
            data = r.json()
        except Exception as e:
            print("buslocationservice API error:", e)
            return None

        # ✅ 공식 예시: 최상단에 msgHeader / msgBody 가 바로 있음
        # 혹시 다른 버전(response 래퍼)도 대응하고 싶으면 분기 처리
        if "response" in data:
            # 다른 API들과 같은 패턴일 수도 있어서 방어적으로 처리
            resp = data.get("response", {})
            header = resp.get("msgHeader", {}) or {}
            body = resp.get("msgBody", {}) or {}
        else:
            header = data.get("msgHeader", {}) or {}
            body = data.get("msgBody", {}) or {}

        query_time = header.get("queryTime", "")
        loc_list = body.get("busLocationList", []) or []

        # 한 대만 있으면 dict, 여러 대면 list → 항상 list 로 맞추기
        if isinstance(loc_list, dict):
            loc_list = [loc_list]

        return query_time, loc_list

    # --------------------
    # 1) GET (노선 전체 버스 위치 목록)
    # --------------------
    if request.method == "GET":
        routeid = request.GET.get("routeid")
        service_date = request.GET.get("service_date")

        if not routeid:
            return JsonResponse(
                {"error": "routeid 파라미터가 필요합니다."},
                status=400,
            )

        # 노선 이름 가져오기 (한 번만)
        URL_ROUTE_INFO = "https://apis.data.go.kr/6410000/busrouteservice/v2/getBusRouteInfoItemv2"
        route_name = ""
        try:
            r_info = requests.get(
                URL_ROUTE_INFO,
                params={
                    "serviceKey": SERVICE_KEY,
                    "routeId": routeid,
                    "format": "json",
                },
                timeout=5,
            )
            info_json = r_info.json()
            route_name = str(
                info_json.get("response", {})
                .get("msgBody", {})
                .get("busRouteInfoItem", {})
                .get("routeName", "")
            )
        except Exception as e:
            print("route info api error:", e)
            route_name = ""

        result = call_buslocation_api(routeid)
        if result is None:
            return JsonResponse(
                {"error": "buslocation api error"},
                status=502,
            )

        query_time, loc_list = result

        out = []
        for item in loc_list:
            try:
                station_seq = int(item.get("stationSeq"))
            except Exception:
                station_seq = None

            remain_raw = item.get("remainSeatCnt")
            try:
                remainseat = (
                    int(remain_raw)
                    if remain_raw not in (None, "", " ", -1)
                    else None
                )
            except Exception:
                remainseat = None

            crowded_raw = item.get("crowded")
            try:
                crowded_level = int(crowded_raw)
                if crowded_level not in (1, 2, 3, 4):
                    raise ValueError
            except Exception:
                # 좌석수 기반 추정
                if remainseat is None:
                    crowded_level = 2
                else:
                    if remainseat >= 35:
                        crowded_level = 1
                    elif remainseat >= 25:
                        crowded_level = 2
                    elif remainseat >= 10:
                        crowded_level = 3
                    else:
                        crowded_level = 4

            service_date_out = service_date or (
                query_time.split(" ")[0] if query_time else ""
            )

            out.append(
                {
                    "service_date": service_date_out,
                    "arrival_time": query_time,
                    "vehid1": str(item.get("vehId") or ""),
                    "station_num": str(station_seq) if station_seq is not None else "",
                    "remainseat_at_arrival": remainseat,
                    "routeid": str(item.get("routeId") or routeid),
                    "routename": route_name,
                    "stationid": str(item.get("stationId") or ""),
                    "crowded_level": crowded_level,
                }
            )

        return JsonResponse(out, safe=False, status=200)

    # --------------------
    # 2) POST (BusSearch 용, 기존 형식 유지)
    # --------------------
    # 요청 바디 파싱
    try:
        body = json.loads(request.body.decode())
    except Exception:
        return JsonResponse({"error": "invalid json body"}, status=400)

    route_id = body.get("routeId")
    stations = body.get("stations", [])

    if not route_id or not stations:
        return JsonResponse({"error": "missing params"}, status=400)

    # 노선의 모든 버스 위치 한 번만 조회
    result = call_buslocation_api(route_id)
    if result is None:
        return JsonResponse(
            {"error": "buslocation api error"},
            status=502,
        )

    query_time, loc_list = result

    # (stationId, stationSeq) 기준으로 인덱싱
    index = {}
    for item in loc_list:
        s_id = str(item.get("stationId"))
        try:
            s_seq = int(item.get("stationSeq"))
        except Exception:
            continue
        key = (s_id, s_seq)
        index.setdefault(key, []).append(item)

    total_stops = len(stations) if stations else 1

    results = []

    for s in stations:
        station_id = s.get("stationId")
        sta_order = s.get("staOrder")

        if not station_id or sta_order is None:
            results.append(
                {
                    "stationId": station_id,
                    "staOrder": sta_order,
                    "raw": None,
                }
            )
            continue

        key = (str(station_id), int(sta_order))
        items_here = index.get(key)
        raw = None

        if items_here:
            item0 = items_here[0]
            try:
                seq = int(item0.get("stationSeq") or sta_order)
            except Exception:
                seq = int(sta_order)

            # BusSearch 에서 쓰던 locationNo1 형식 맞추기:
            #   totalStops - 1 - locationNo1 = 타임라인 index
            # → 우리가 그냥 "해당 정류장 index" 에 꽂히도록 역산
            location_no1 = max(total_stops - 1 - (seq - 1), 0)

            remain_raw = item0.get("remainSeatCnt")
            try:
                remain_seat = (
                    int(remain_raw)
                    if remain_raw not in (None, "", " ", -1)
                    else None
                )
            except Exception:
                remain_seat = None

            raw = {
                "vehId1": str(item0.get("vehId") or ""),
                "locationNo1": location_no1,
                "remainSeatCnt1": remain_seat,
                "crowded1": item0.get("crowded"),
                "queryTime": query_time,
            }

        results.append(
            {
                "stationId": station_id,
                "staOrder": sta_order,
                "raw": raw,
            }
        )

    return JsonResponse(
        {
            "routeId": route_id,
            "results": results,
        },
        status=200,
    )


# -----------------------------
#  station_realtime (정류장 화면용)
# -----------------------------
@csrf_exempt
@require_GET

def station_realtime(request):
    # # 🔥 개발 모드: 외부 API 안 쓰고, 빈 배열만 돌려줘서 502 막기
    # if USE_FAKE_REALTIME:
    #     return JsonResponse([], safe=False, status=200)

    stationid = request.GET.get("stationid")
    service_date = request.GET.get("service_date")  # 그대로 돌려만 줌
    time_slot = request.GET.get("time_slot")  # 지금은 따로 쓰진 않음

    if not stationid:
        return JsonResponse(
            {"error": "stationid 파라미터가 필요합니다."},
            status=400,
        )

    stationid = request.GET.get("stationid")
    service_date = request.GET.get("service_date")
    time_slot = request.GET.get("time_slot")

    if not stationid:
        return JsonResponse(
            {"error": "stationid 파라미터가 필요합니다."},
            status=400,
        )

    # ✅ 외부 API 대신 local JSON 사용
    local_routes = get_local_routes_via_station(stationid)
    # local_routes: [{ "routeId": ..., "routeName": ..., "staOrder": ... }, ...]

    if not local_routes:
        return JsonResponse([], safe=False, status=200)

    URL_ARRIVAL = (
        "https://apis.data.go.kr/6410000/busarrivalservice/v2/getBusArrivalItemv2"
    )

    results = []

    for route in local_routes:
        routeid = str(route.get("routeId"))
        routename = str(route.get("routeName"))
        sta_order = route.get("staOrder")

        if not routeid or sta_order is None:
            continue

        try:
            r2 = requests.get(
                URL_ARRIVAL,
                params={
                    "serviceKey": SERVICE_KEY,
                    "stationId": stationid,
                    "routeId": routeid,
                    "staOrder": sta_order,
                    "format": "json",
                },
                timeout=5,
            )
            arrival_json = r2.json()
        except Exception:
            continue

        resp = arrival_json.get("response", {})
        header = resp.get("msgHeader", {}) or {}
        body_item = resp.get("msgBody", {}).get("busArrivalItem")

        if not body_item:
            continue

        query_time = header.get("queryTime", "")

        vehid1 = str(
            body_item.get("vehId1")
            or body_item.get("vehid1")
            or ""
        )

        remain_raw = body_item.get("remainSeatCnt1")
        try:
            remainseat = (
                int(remain_raw)
                if remain_raw not in (None, "", " ")
                else None
            )
        except Exception:
            remainseat = None

        crowded_raw = body_item.get("crowded1")

        try:
            crowded_level = int(crowded_raw)
            if crowded_level not in (1, 2, 3, 4):
                raise ValueError
        except Exception:
            if remainseat is None:
                crowded_level = 2
            else:
                if remainseat >= 35:
                    crowded_level = 1
                elif remainseat >= 25:
                    crowded_level = 2
                elif remainseat >= 10:
                    crowded_level = 3
                else:
                    crowded_level = 4

        results.append(
            {
                "service_date": service_date
                or (query_time.split(" ")[0] if query_time else ""),
                "arrival_time": query_time,
                "vehid1": vehid1,
                "station_num": str(sta_order),
                "remainseat_at_arrival": remainseat,
                "routeid": routeid,
                "routename": routename,
                "stationid": stationid,
                "crowded_level": crowded_level,
            }
        )

    return JsonResponse(results, safe=False, status=200)


# -----------------------------
#  recommend_route (임시 더미)
# -----------------------------
@csrf_exempt
@require_GET
def recommend_route(request):
    """
    경로 추천 API (아직 더미)
    """
    origin_stationid = request.GET.get("origin_stationid")
    dest_stationid = request.GET.get("dest_stationid")
    weekday = request.GET.get("weekday", "월요일")
    time_slot = request.GET.get("time_slot", "8:30")
    time_type = request.GET.get("time_type", "도착시간")
    fast_option = request.GET.get("fast_option", "최단시간")

    if not origin_stationid or not dest_stationid:
        return JsonResponse(
            {"ok": False, "error": "origin_stationid와 dest_stationid 파라미터가 필요합니다."},
            status=400,
        )

    try:
        data = {
            "ok": True,
            "origin_stationid": origin_stationid,
            "dest_stationid": dest_stationid,
            "weekday": weekday,
            "time_slot": time_slot,
            "time_type": time_type,
            "fast_option": fast_option,
            "recommended_route": {
                "bus_numbers": [],
                "routeid": None,
                "duration_minutes": None,
                "congestion_level": None,
            },
            "message": "경로 추천 기능은 현재 개발 중입니다.",
        }

        return JsonResponse(data, status=200)
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": f"서버 오류: {str(e)}"},
            status=500,
        )