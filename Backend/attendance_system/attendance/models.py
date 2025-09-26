# attendance/models.py
from django.db import models
from users.models import Student, Teacher, Department, AcademicYear
import uuid

class Subject(models.Model):
    subject_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, unique=True)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    semester = models.IntegerField()
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)
    
    # Subject Details
    credits = models.IntegerField(default=3)
    subject_type = models.CharField(max_length=20, choices=[
        ('theory', 'Theory'),
        ('practical', 'Practical'),
        ('project', 'Project')
    ], default='theory')
    
    # Class Schedule
    total_classes = models.IntegerField(default=60)  # Total classes per semester
    
    # Relationships
    teachers = models.ManyToManyField(Teacher, through='SubjectTeacher')
    students = models.ManyToManyField(Student, through='SubjectEnrollment')
    
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.code})"

class SubjectTeacher(models.Model):
    """Through model for Subject-Teacher relationship"""
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    is_primary = models.BooleanField(default=False)  # Primary teacher for the subject
    assigned_date = models.DateField(auto_now_add=True)
    
    class Meta:
        unique_together = ['subject', 'teacher']

class SubjectEnrollment(models.Model):
    """Through model for Subject-Student enrollment"""
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    enrollment_date = models.DateField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ['subject', 'student']

class ClassSchedule(models.Model):
    """Class scheduling model"""
    schedule_id = models.UUIDField(default=uuid.uuid4, unique=True)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    
    # Schedule Details
    day_of_week = models.CharField(max_length=10, choices=[
        ('monday', 'Monday'),
        ('tuesday', 'Tuesday'),
        ('wednesday', 'Wednesday'),
        ('thursday', 'Thursday'),
        ('friday', 'Friday'),
        ('saturday', 'Saturday'),
    ])
    start_time = models.TimeField()
    end_time = models.TimeField()
    room_number = models.CharField(max_length=20, blank=True)
    
    # Academic Context
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)
    semester = models.IntegerField()
    section = models.CharField(max_length=5)
    
    is_active = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ['subject', 'day_of_week', 'start_time', 'academic_year', 'semester', 'section']

class AttendanceSession(models.Model):
    """Attendance session for a specific class"""
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    schedule = models.ForeignKey(ClassSchedule, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Session Details
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField(blank=True, null=True)
    session_type = models.CharField(max_length=20, choices=[
        ('regular', 'Regular Class'),
        ('extra', 'Extra Class'),
        ('makeup', 'Makeup Class'),
        ('exam', 'Exam')
    ], default='regular')
    
    # Attendance Tracking
    total_enrolled = models.IntegerField(default=0)
    total_present = models.IntegerField(default=0)
    total_absent = models.IntegerField(default=0)
    total_late = models.IntegerField(default=0)
    
    # Session Status
    is_active = models.BooleanField(default=True)
    is_completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.subject.name} - {self.date} ({self.start_time})"

class Attendance(models.Model):
    STATUS_CHOICES = (
        ('present', 'Present'),
        ('absent', 'Absent'),
        ('late', 'Late'),
    )
    
    attendance_id = models.UUIDField(default=uuid.uuid4, unique=True)
    session = models.ForeignKey(AttendanceSession, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    
    # Attendance Details
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    marked_at = models.DateTimeField(auto_now_add=True)
    marked_by_teacher = models.ForeignKey(Teacher, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Face Recognition Details
    face_match_confidence = models.FloatField(blank=True, null=True)  # Face recognition confidence score
    device_used = models.ForeignKey('devices.Device', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Additional Information
    remarks = models.TextField(blank=True)
    is_manual_entry = models.BooleanField(default=False)  # Manual vs automatic marking
    
    class Meta:
        unique_together = ['session', 'student']
    
    def __str__(self):
        return f"{self.student} - {self.session.subject} on {self.session.date}: {self.status}"
