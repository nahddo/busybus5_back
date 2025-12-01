from django.contrib import admin
from .models import bus_arrival_past, Favorite, SavedRoute

@admin.action(description="버스 좌석 예측 모델 학습 실행")
def run_training_action(modeladmin, request, queryset):
    modeladmin.message_user(request, "모델 학습이 완료되었습니다!")


@admin.register(bus_arrival_past)
class BusArrivalPastAdmin(admin.ModelAdmin):
    # 이건 네 테이블 필드에 맞게 적당히
    list_display = ("routeid", "timestamp", "remainseatcnt1", "vehid1", "station_num")
    actions = [run_training_action]


@admin.register(Favorite)
class FavoriteAdmin(admin.ModelAdmin):
    list_display = ("user", "label", "type", "created_at")
    list_filter = ("type", "created_at")
    search_fields = ("user__username", "label")


@admin.register(SavedRoute)
class SavedRouteAdmin(admin.ModelAdmin):
    list_display = ("user", "from_location", "to_location", "type", "created_at")
    list_filter = ("type", "created_at")
    search_fields = ("user__username", "from_location", "to_location")