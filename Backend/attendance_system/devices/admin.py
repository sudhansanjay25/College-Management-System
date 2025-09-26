# devices/admin.py
from django.contrib import admin
from .models import Device, DeviceLog

@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ['device_id', 'name', 'device_type', 'location', 'is_online', 'is_active', 'last_heartbeat']
    list_filter = ['device_type', 'is_online', 'is_active', 'department']
    search_fields = ['device_id', 'name', 'location', 'ip_address']
    readonly_fields = ['installed_date', 'last_heartbeat']

@admin.register(DeviceLog)
class DeviceLogAdmin(admin.ModelAdmin):
    list_display = ['device', 'log_type', 'message', 'timestamp']
    list_filter = ['log_type', 'timestamp']
    search_fields = ['device__device_id', 'message']
    readonly_fields = ['timestamp']
