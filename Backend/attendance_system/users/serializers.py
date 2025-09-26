# users/serializers.py
from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import (User, Student, Teacher, Admin, Department, AcademicYear, 
                    Subject, ExamType, StudentMarks, SemesterResult, StudentCGPA)

class DepartmentSerializer(serializers.ModelSerializer):
    # head_of_department_name = serializers.CharField(source='head_of_department.full_name', read_only=True)
    
    class Meta:
        model = Department
        fields = '__all__'

class AcademicYearSerializer(serializers.ModelSerializer):
    class Meta:
        model = AcademicYear
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'user_type', 'is_active', 'password']
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        return user

class StudentSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    department_name = serializers.CharField(source='department.name', read_only=True)
    academic_year_display = serializers.CharField(source='academic_year.year', read_only=True)
    full_name = serializers.CharField(read_only=True)
    
    class Meta:
        model = Student
        fields = '__all__'

class StudentCreateSerializer(serializers.ModelSerializer):
    user_data = UserSerializer(write_only=True)
    
    class Meta:
        model = Student
        fields = ['student_id', 'first_name', 'last_name', 'email', 'phone_number',
                 'department', 'academic_year', 'current_semester', 'section',
                 'date_of_birth', 'gender', 'address', 'parent_name', 'parent_phone',
                 'parent_email', 'profile_picture', 'user_data']
    
    def create(self, validated_data):
        user_data = validated_data.pop('user_data')
        user_data['user_type'] = 'student'
        user_data['username'] = validated_data['student_id']
        
        user_serializer = UserSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save()
        
        student = Student.objects.create(user=user, **validated_data)
        return student

class TeacherSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    department_name = serializers.CharField(source='department.name', read_only=True)
    full_name = serializers.CharField(read_only=True)
    
    class Meta:
        model = Teacher
        fields = '__all__'

class TeacherCreateSerializer(serializers.ModelSerializer):
    user_data = UserSerializer(write_only=True)
    
    class Meta:
        model = Teacher
        fields = ['teacher_id', 'first_name', 'last_name', 'email', 'phone_number',
                 'department', 'designation', 'specialization', 'qualification',
                 'experience_years', 'profile_picture', 'user_data']
    
    def create(self, validated_data):
        user_data = validated_data.pop('user_data')
        user_data['user_type'] = 'teacher'
        user_data['username'] = validated_data['teacher_id']
        
        user_serializer = UserSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save()
        
        teacher = Teacher.objects.create(user=user, **validated_data)
        return teacher

class AdminSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Admin
        fields = '__all__'

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()
    
    def validate(self, data):
        username = data.get('username')
        password = data.get('password')
        
        if username and password:
            user = authenticate(username=username, password=password)
            if user:
                if not user.is_active:
                    raise serializers.ValidationError('User account is disabled.')
                data['user'] = user
            else:
                raise serializers.ValidationError('Invalid credentials.')
        else:
            raise serializers.ValidationError('Username and password required.')
        
        return data


class PasswordChangeSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True, required=True)
    new_password = serializers.CharField(write_only=True, required=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True, required=True)

    def validate(self, attrs):
        user = self.context['request'].user
        old = attrs.get('old_password')
        new = attrs.get('new_password')
        confirm = attrs.get('confirm_password')

        if not user.check_password(old):
            raise serializers.ValidationError({'old_password': 'Old password is incorrect'})
        if new != confirm:
            raise serializers.ValidationError({'confirm_password': 'Passwords do not match'})
        if old == new:
            raise serializers.ValidationError({'new_password': 'New password must be different from old password'})
        # Basic hardness checks (can be replaced with Django password validators)
        if len(new) < 8:
            raise serializers.ValidationError({'new_password': 'Password must be at least 8 characters'})
        return attrs

    def save(self, **kwargs):
        user = self.context['request'].user
        new = self.validated_data['new_password']
        user.set_password(new)
        user.must_change_password = False
        user.save(update_fields=['password', 'must_change_password'])
        return user


class SubjectSerializer(serializers.ModelSerializer):
    department_name = serializers.CharField(source='department.name', read_only=True)
    
    class Meta:
        model = Subject
        fields = '__all__'

class ExamTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExamType
        fields = '__all__'

class StudentMarksSerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    student_id = serializers.CharField(source='student.student_id', read_only=True)
    subject_name = serializers.CharField(source='subject.name', read_only=True)
    subject_code = serializers.CharField(source='subject.code', read_only=True)
    exam_type_name = serializers.CharField(source='exam_type.display_name', read_only=True)
    entered_by_name = serializers.CharField(source='entered_by.full_name', read_only=True)
    percentage = serializers.ReadOnlyField()
    
    class Meta:
        model = StudentMarks
        fields = '__all__'
    
    def create(self, validated_data):
        # Set the teacher who entered the marks
        if self.context['request'].user.user_type == 'teacher':
            try:
                teacher = Teacher.objects.get(user=self.context['request'].user)
                validated_data['entered_by'] = teacher
            except Teacher.DoesNotExist:
                pass
        
        return super().create(validated_data)

class StudentMarksDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for student to view their own marks"""
    subject = SubjectSerializer(read_only=True)
    exam_type = ExamTypeSerializer(read_only=True)
    percentage = serializers.ReadOnlyField()
    
    class Meta:
        model = StudentMarks
        fields = ['subject', 'exam_type', 'marks_obtained', 'max_marks', 'percentage', 'entered_date']

class SemesterResultSerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    student_id = serializers.CharField(source='student.student_id', read_only=True)
    academic_year_display = serializers.CharField(source='academic_year.year', read_only=True)
    
    class Meta:
        model = SemesterResult
        fields = '__all__'

class StudentCGPASerializer(serializers.ModelSerializer):
    student_name = serializers.CharField(source='student.full_name', read_only=True)
    student_id = serializers.CharField(source='student.student_id', read_only=True)
    
    class Meta:
        model = StudentCGPA
        fields = '__all__'

class StudentMarksReportSerializer(serializers.Serializer):
    """Serializer for student's complete marks report"""
    student = StudentSerializer(read_only=True)
    semester_marks = serializers.SerializerMethodField()
    semester_results = SemesterResultSerializer(many=True, read_only=True)
    cgpa_record = StudentCGPASerializer(read_only=True)
    
    def get_semester_marks(self, obj):
        """Get marks grouped by semester and exam type"""
        marks = StudentMarks.objects.filter(student=obj).select_related(
            'subject', 'exam_type', 'academic_year'
        ).order_by('subject__semester', 'exam_type__name', 'subject__name')
        
        semester_data = {}
        for mark in marks:
            sem = mark.subject.semester
            if sem not in semester_data:
                semester_data[sem] = {
                    'internal1': [],
                    'internal2': [],
                    'practical': [],
                    'semester': []
                }
            
            exam_type = mark.exam_type.name
            semester_data[sem][exam_type].append(StudentMarksDetailSerializer(mark).data)
        
        return semester_data
