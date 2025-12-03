from django.urls import path
from .views import (
    run_training,
    predict_seat,
    bus_realtime,
    station_realtime,
    recommend_route,
)
from .views_auth import signup, login_view, logout_view, current_user
from .views_user_data import favorites, favorite_detail, saved_routes, saved_route_detail

urlpatterns = [
    path('auth/signup/', signup),
    path('auth/login/', login_view),
    path('auth/logout/', logout_view),
    path('auth/me/', current_user),

    path('predict-seat/', predict_seat, name='predict_seat'),
    path('train/', run_training, name='run_training'),

    # 실시간 데이터 API
    path('bus/realtime/', bus_realtime, name='bus_realtime'),
    path('station/realtime/', station_realtime, name='station_realtime'),

    # 경로 추천
    path('recommend-route/', recommend_route, name='recommend_route'),

    # 사용자 데이터 API
    path('favorites/', favorites, name='favorites'),
    path('favorites/<int:favorite_id>/', favorite_detail, name='favorite_detail'),
    path('saved-routes/', saved_routes, name='saved_routes'),
    path('saved-routes/<int:route_id>/', saved_route_detail, name='saved_route_detail'),
]