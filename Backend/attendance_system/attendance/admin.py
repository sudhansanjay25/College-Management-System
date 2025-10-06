# attendance/admin.py
from django.contrib import admin
from django import forms
from django.forms.widgets import SelectDateWidget
from django.utils import timezone
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

    class AttendanceSessionForm(forms.ModelForm):
        class Meta:
            model = AttendanceSession
            fields = '__all__'
            widgets = {
                # Allow easy selection for sessions across past year and next year
                'date': SelectDateWidget(
                    years=range(timezone.now().year + 1, timezone.now().year - 1, -1)
                )
            }

    form = AttendanceSessionForm

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'session', 'status', 'marked_at', 'face_match_confidence']
    list_filter = ['status', 'session__date', 'is_manual_entry']
    search_fields = ['student__student_id', 'student__first_name', 'session__subject__name']
    readonly_fields = ['marked_at']
