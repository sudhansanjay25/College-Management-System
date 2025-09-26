# devices/models.py
from django.db import models
from users.models import Department
import uuid

class Device(models.Model):
    DEVICE_TYPES = [
        ('kiosk', 'Attendance Kiosk'),
        ('camera', 'IP Camera'),
        ('tablet', 'Tablet Device'),
    ]
    
    device_id = models.CharField(max_length=20, unique=True)
    device_uuid = models.UUIDField(default=uuid.uuid4, unique=True)
    name = models.CharField(max_length=100)
    device_type = models.CharField(max_length=20, choices=DEVICE_TYPES, default='kiosk')
    
    # Location Details
    location = models.CharField(max_length=100)
    room_number = models.CharField(max_length=20, blank=True)
    building = models.CharField(max_length=50, blank=True)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Network Details
    ip_address = models.GenericIPAddressField()
    mac_address = models.CharField(max_length=17, blank=True)  # MAC address format: XX:XX:XX:XX:XX:XX
    
    # Device Status
    is_online = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    last_heartbeat = models.DateTimeField(blank=True, null=True)
    
    # Configuration
    settings = models.JSONField(default=dict, blank=True)  # Device-specific settings
    
    # Maintenance
    installed_date = models.DateField(auto_now_add=True)
    last_maintenance = models.DateField(blank=True, null=True)
    
    def __str__(self):
        return f"Device {self.device_id} at {self.location}"

class DeviceLog(models.Model):
    """Device activity and error logs"""
    LOG_TYPES = [
        ('info', 'Information'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('attendance', 'Attendance Event'),
    ]
    
    device = models.ForeignKey(Device, on_delete=models.CASCADE)
    log_type = models.CharField(max_length=20, choices=LOG_TYPES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    additional_data = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.device.device_id} - {self.log_type} - {self.timestamp}"