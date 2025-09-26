# attendance/admin.py
from django.contrib import admin
from .models import Subject, SubjectTeacher, SubjectEnrollment, ClassSchedule, AttendanceSession, Attendance

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['subject_id', 'name', 'code', 'department', 'semester', 'credits', 'is_active']
    list_filter = ['department', 'semester', 'subject_type', 'is_active', 'academic_year']
    search_fields = ['name', 'code', 'subject_id']

@admin.register(ClassSchedule)
class ClassScheduleAdmin(admin.ModelAdmin):
    list_display = ['subject', 'teacher', 'day_of_week', 'start_time', 'end_time', 'room_number']
    list_filter = ['day_of_week', 'semester', 'academic_year']
    search_fields = ['subject__name', 'teacher__first_name', 'room_number']

@admin.register(AttendanceSession)
class AttendanceSessionAdmin(admin.ModelAdmin):
    list_display = ['subject', 'teacher', 'date', 'start_time', 'total_enrolled', 'total_present', 'is_completed']
    list_filter = ['date', 'session_type', 'is_completed']
    search_fields = ['subject__name', 'teacher__first_name']
    readonly_fields = ['created_at', 'completed_at']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'session', 'status', 'marked_at', 'face_match_confidence']
    list_filter = ['status', 'session__date', 'is_manual_entry']
    search_fields = ['student__student_id', 'student__first_name', 'session__subject__name']
    readonly_fields = ['marked_at']
