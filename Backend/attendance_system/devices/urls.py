# devices/urls.py
from django.urls import path
from . import views

app_name = 'devices'

urlpatterns = [
    # Device Management
    path('', views.DeviceListCreateView.as_view(), name='device-list'),
    path('<int:pk>/', views.DeviceDetailView.as_view(), name='device-detail'),
    
    # Device Operations
    path('<str:device_id>/heartbeat/', views.device_heartbeat, name='device-heartbeat'),
    path('<str:device_id>/log/', views.device_log_event, name='device-log'),
    path('<str:device_id>/reset/', views.reset_device, name='device-reset'),
    
    # Device Monitoring
    path('logs/', views.DeviceLogListView.as_view(), name='device-logs'),
    path('dashboard/', views.device_dashboard, name='device-dashboard'),
]