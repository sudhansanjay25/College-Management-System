# devices/serializers.py
from rest_framework import serializers
from .models import Device, DeviceLog

class DeviceSerializer(serializers.ModelSerializer):
    department_name = serializers.CharField(source='department.name', read_only=True)
    status_display = serializers.SerializerMethodField()
    last_seen = serializers.SerializerMethodField()
    
    class Meta:
        model = Device
        fields = '__all__'
    
    def get_status_display(self, obj):
        return 'Online' if obj.is_online else 'Offline'
    
    def get_last_seen(self, obj):
        if obj.last_heartbeat:
            return obj.last_heartbeat.strftime('%Y-%m-%d %H:%M:%S')
        return 'Never'

class DeviceLogSerializer(serializers.ModelSerializer):
    device_name = serializers.CharField(source='device.name', read_only=True)
    device_id_display = serializers.CharField(source='device.device_id', read_only=True)
    
    class Meta:
        model = DeviceLog
        fields = '__all__'

class DeviceCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = ['device_id', 'name', 'device_type', 'location', 'room_number', 
                 'building', 'department', 'ip_address', 'mac_address', 'settings']
