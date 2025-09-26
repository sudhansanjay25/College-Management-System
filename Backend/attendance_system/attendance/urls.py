# attendance/urls.py
from django.urls import path
from . import views

app_name = 'attendance'

urlpatterns = [
    # Subjects
    path('subjects/', views.SubjectListCreateView.as_view(), name='subject-list'),
    path('subjects/<int:pk>/', views.SubjectDetailView.as_view(), name='subject-detail'),
    
    # Class Schedules
    path('schedules/', views.ClassScheduleListCreateView.as_view(), name='schedule-list'),
    
    # Attendance Sessions
    path('sessions/', views.AttendanceSessionListCreateView.as_view(), name='session-list'),
    path('sessions/<int:pk>/', views.AttendanceSessionDetailView.as_view(), name='session-detail'),
    path('sessions/start/', views.start_attendance_session, name='start-session'),
    path('sessions/<int:session_id>/complete/', views.complete_attendance_session, name='complete-session'),
    
    # Attendance Marking
    path('mark-face/', views.mark_attendance_face, name='mark-face'),
    
    # Reports
    path('reports/', views.attendance_report, name='attendance-report'),
]
