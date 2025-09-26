# devices/views.py
from django.db import models
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.utils import timezone
from datetime import timedelta
from .models import Device, DeviceLog
from .serializers import DeviceSerializer, DeviceLogSerializer, DeviceCreateSerializer

class DeviceListCreateView(generics.ListCreateAPIView):
    queryset = Device.objects.filter(is_active=True)
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return DeviceCreateSerializer
        return DeviceSerializer
    
    def get_queryset(self):
        queryset = Device.objects.filter(is_active=True)
        device_type = self.request.query_params.get('type')
        department = self.request.query_params.get('department')
        is_online = self.request.query_params.get('is_online')
        
        if device_type:
            queryset = queryset.filter(device_type=device_type)
        if department:
            queryset = queryset.filter(department=department)
        if is_online is not None:
            queryset = queryset.filter(is_online=is_online.lower() == 'true')
            
        return queryset.order_by('location', 'device_id')

class DeviceDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Device.objects.all()
    serializer_class = DeviceSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def perform_destroy(self, instance):
        instance.is_active = False
        instance.save()

class DeviceLogListView(generics.ListAPIView):
    queryset = DeviceLog.objects.all()
    serializer_class = DeviceLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = DeviceLog.objects.all()
        device_id = self.request.query_params.get('device_id')
        log_type = self.request.query_params.get('log_type')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        
        if device_id:
            queryset = queryset.filter(device__device_id=device_id)
        if log_type:
            queryset = queryset.filter(log_type=log_type)
        if start_date:
            queryset = queryset.filter(timestamp__gte=start_date)
        if end_date:
            queryset = queryset.filter(timestamp__lte=end_date)
            
        return queryset.order_by('-timestamp')

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def device_heartbeat(request, device_id):
    """Update device heartbeat and status"""
    try:
        device = get_object_or_404(Device, device_id=device_id)
        
        # Update device status
        device.is_online = True
        device.last_heartbeat = timezone.now()
        device.save()
        
        # Log heartbeat
        DeviceLog.objects.create(
            device=device,
            log_type='info',
            message='Device heartbeat received',
            additional_data=request.data
        )
        
        return Response({
            'message': 'Heartbeat received',
            'device_status': 'online',
            'timestamp': timezone.now().isoformat()
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def device_log_event(request, device_id):
    """Log device events"""
    try:
        device = get_object_or_404(Device, device_id=device_id)
        log_type = request.data.get('log_type', 'info')
        message = request.data.get('message', '')
        additional_data = request.data.get('additional_data', {})
        
        DeviceLog.objects.create(
            device=device,
            log_type=log_type,
            message=message,
            additional_data=additional_data
        )
        
        return Response({'message': 'Event logged successfully'}, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def device_dashboard(request):
    """Get device dashboard statistics"""
    try:
        total_devices = Device.objects.filter(is_active=True).count()
        online_devices = Device.objects.filter(is_active=True, is_online=True).count()
        offline_devices = total_devices - online_devices
        
        # Devices that haven't sent heartbeat in last 5 minutes
        five_minutes_ago = timezone.now() - timedelta(minutes=5)
        stale_devices = Device.objects.filter(
            is_active=True,
            last_heartbeat__lt=five_minutes_ago
        ).count()
        
        # Recent errors
        recent_errors = DeviceLog.objects.filter(
            log_type='error',
            timestamp__gte=timezone.now() - timedelta(hours=24)
        ).count()
        
        # Device type breakdown
        device_types = Device.objects.filter(is_active=True).values('device_type').annotate(
            count=models.Count('id')
        )
        
        return Response({
            'total_devices': total_devices,
            'online_devices': online_devices,
            'offline_devices': offline_devices,
            'stale_devices': stale_devices,
            'recent_errors': recent_errors,
            'device_types': list(device_types)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def reset_device(request, device_id):
    """Reset device (mark as offline and clear settings)"""
    try:
        device = get_object_or_404(Device, device_id=device_id)
        
        device.is_online = False
        device.last_heartbeat = None
        # Optionally reset settings
        if request.data.get('reset_settings', False):
            device.settings = {}
        device.save()
        
        DeviceLog.objects.create(
            device=device,
            log_type='info',
            message='Device reset by admin',
            additional_data={'reset_by': request.user.username}
        )
        
        return Response({'message': 'Device reset successfully'}, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
