# attendance/serializers.py
from rest_framework import serializers
from .models import Subject, SubjectTeacher, SubjectEnrollment, ClassSchedule, AttendanceSession, Attendance
from users.serializers import StudentSerializer, TeacherSerializer

class SubjectSerializer(serializers.ModelSerializer):
    department_name = serializers.CharField(source='department.name', read_only=True)
    academic_year_display = serializers.CharField(source='academic_year.year', read_only=True)
    teacher_count = serializers.SerializerMethodField()
    student_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Subject
        fields = '__all__'
    
    def get_teacher_count(self, obj):
        return obj.teachers.count()
    
    def get_student_count(self, obj):
        return obj.students.count()

class SubjectTeacherSerializer(serializers.ModelSerializer):
    teacher_name = serializers.CharField(source='teacher.full_name', read_only=True)
    subject_name = serializers.CharField(source='subject.name', read_only=True)
    
    class Meta:
        model = SubjectTeacher
        fields = '__all__'

class SubjectEnrollmentSerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    student_id = serializers.CharField(source='student.student_id', read_only=True)
    subject_name = serializers.CharField(source='subject.name', read_only=True)
    
    class Meta:
        model = SubjectEnrollment
        fields = '__all__'

class ClassScheduleSerializer(serializers.ModelSerializer):
    subject_name = serializers.CharField(source='subject.name', read_only=True)
    teacher_name = serializers.CharField(source='teacher.full_name', read_only=True)
    academic_year_display = serializers.CharField(source='academic_year.year', read_only=True)
    
    class Meta:
        model = ClassSchedule
        fields = '__all__'

class AttendanceSessionSerializer(serializers.ModelSerializer):
    subject_name = serializers.CharField(source='subject.name', read_only=True)
    teacher_name = serializers.CharField(source='teacher.full_name', read_only=True)
    attendance_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = AttendanceSession
        fields = '__all__'
    
    def get_attendance_percentage(self, obj):
        if obj.total_enrolled > 0:
            return round((obj.total_present / obj.total_enrolled) * 100, 2)
        return 0

class AttendanceSerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    student_id = serializers.CharField(source='student.student_id', read_only=True)
    subject_name = serializers.CharField(source='session.subject.name', read_only=True)
    session_date = serializers.DateField(source='session.date', read_only=True)
    session_time = serializers.TimeField(source='session.start_time', read_only=True)
    
    class Meta:
        model = Attendance
        fields = '__all__'

class AttendanceCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = ['session', 'student', 'status', 'face_match_confidence', 'device_used', 'remarks']

class AttendanceReportSerializer(serializers.Serializer):
    student = StudentSerializer()
    total_classes = serializers.IntegerField()
    present_count = serializers.IntegerField()
    absent_count = serializers.IntegerField()
    late_count = serializers.IntegerField()
    attendance_percentage = serializers.FloatField()