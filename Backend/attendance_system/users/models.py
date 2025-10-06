# users/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid

class User(AbstractUser):
    USER_TYPES = (
        ('admin', 'Admin'),
        ('teacher', 'Teacher'),
        ('student', 'Student'),
    )
    user_type = models.CharField(max_length=10, choices=USER_TYPES)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Force user to change password on next authenticated action (primarily after admin-created default credentials)
    must_change_password = models.BooleanField(default=False)

class Department(models.Model):
    """Department model to centralize department management"""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=10, unique=True)
    # head_of_department = models.ForeignKey('Teacher', on_delete=models.SET_NULL, null=True, blank=True, related_name='headed_department')
    established_date = models.DateField(blank=True, null=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.code})"

class AcademicYear(models.Model):
    """Academic Year model for proper year management"""
    year = models.CharField(max_length=20, unique=True)  # e.g., "2024-2025"
    start_date = models.DateField()
    end_date = models.DateField()
    is_current = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.year

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    student_id = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    
    # Academic Information
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)
    current_semester = models.IntegerField()
    section = models.CharField(max_length=5)  # A, B, C, etc.
    
    # Personal Information
    date_of_birth = models.DateField(blank=True, null=True)
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], blank=True)
    address = models.TextField(blank=True)
    
    # Guardian Information
    parent_name = models.CharField(max_length=100, blank=True)
    parent_phone = models.CharField(max_length=15, blank=True)
    parent_email = models.EmailField(blank=True)
    
    # System Information
    enrollment_date = models.DateField(auto_now_add=True)
    graduation_date = models.DateField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    
    # Face Recognition
    face_encoding = models.BinaryField(blank=True, null=True)
    profile_picture = models.ImageField(upload_to='student_profiles/', blank=True, null=True)
    
    class Meta:
        unique_together = ['department', 'academic_year', 'current_semester', 'section', 'student_id']
    
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.student_id})"
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

class Teacher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    teacher_id = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    
    # Professional Information
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    designation = models.CharField(max_length=50)  # Professor, Associate Professor, etc.
    specialization = models.CharField(max_length=100, blank=True)
    qualification = models.CharField(max_length=100, blank=True)
    experience_years = models.IntegerField(default=0)
    
    # System Information
    joining_date = models.DateField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    # Face Recognition
    face_encoding = models.BinaryField(blank=True, null=True)
    profile_picture = models.ImageField(upload_to='teacher_profiles/', blank=True, null=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.designation})"
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

class Subject(models.Model):
    """Subject model for each department and semester"""
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20, unique=True)
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='subjects')
    semester = models.IntegerField()
    credits = models.DecimalField(max_digits=3, decimal_places=1)  # e.g., 3.0, 2.5
    has_practical = models.BooleanField(default=False)  # Whether this subject has practical exams
    is_active = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ['department', 'semester', 'code']
    
    def __str__(self):
        return f"{self.name} ({self.code}) - Sem {self.semester}"

class ExamType(models.Model):
    """Types of exams"""
    EXAM_CHOICES = [
        ('internal1', 'Internal Exam 1'),
        ('internal2', 'Internal Exam 2'),
        ('practical', 'Practical Exam'),
        ('semester', 'Semester End Exam'),
    ]
    
    name = models.CharField(max_length=20, choices=EXAM_CHOICES, unique=True)
    display_name = models.CharField(max_length=50)
    max_marks = models.IntegerField(default=100)
    weightage_percentage = models.DecimalField(max_digits=5, decimal_places=2, default=100.00)  # For SGPA calculation
    
    def __str__(self):
        return self.display_name

class StudentMarks(models.Model):
    """Student marks for different exams"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='marks')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    exam_type = models.ForeignKey(ExamType, on_delete=models.CASCADE)
    marks_obtained = models.DecimalField(max_digits=5, decimal_places=2)
    max_marks = models.IntegerField(default=100)
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)
    entered_by = models.ForeignKey(Teacher, on_delete=models.SET_NULL, null=True, blank=True)
    entered_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['student', 'subject', 'exam_type', 'academic_year']
    
    def __str__(self):
        return f"{self.student.full_name} - {self.subject.name} - {self.exam_type.display_name}: {self.marks_obtained}/{self.max_marks}"
    
    @property
    def percentage(self):
        return (self.marks_obtained / self.max_marks) * 100

class SemesterResult(models.Model):
    """Semester-wise SGPA calculation for students"""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='semester_results')
    semester = models.IntegerField()
    academic_year = models.ForeignKey(AcademicYear, on_delete=models.CASCADE)
    sgpa = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    total_credits = models.DecimalField(max_digits=5, decimal_places=1, default=0)
    calculated_date = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['student', 'semester', 'academic_year']
    
    def __str__(self):
        return f"{self.student.full_name} - Sem {self.semester} - SGPA: {self.sgpa}"
    
    def calculate_sgpa(self):
        """Calculate SGPA based on semester end exam marks"""
        semester_marks = StudentMarks.objects.filter(
            student=self.student,
            subject__semester=self.semester,
            exam_type__name='semester',
            academic_year=self.academic_year
        ).select_related('subject')
        
        total_grade_points = 0
        total_credits = 0
        
        for mark in semester_marks:
            # Grade point calculation (10-point scale)
            percentage = mark.percentage
            if percentage >= 90:
                grade_point = 10
            elif percentage >= 80:
                grade_point = 9
            elif percentage >= 70:
                grade_point = 8
            elif percentage >= 60:
                grade_point = 7
            elif percentage >= 50:
                grade_point = 6
            elif percentage >= 40:
                grade_point = 5
            else:
                grade_point = 0
            
            total_grade_points += grade_point * float(mark.subject.credits)
            total_credits += float(mark.subject.credits)
        
        if total_credits > 0:
            self.sgpa = total_grade_points / total_credits
            self.total_credits = total_credits
            self.save()
        
        return self.sgpa

class StudentCGPA(models.Model):
    """Overall CGPA calculation for students"""
    student = models.OneToOneField(Student, on_delete=models.CASCADE, related_name='cgpa_record')
    cgpa = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    total_credits = models.DecimalField(max_digits=6, decimal_places=1, default=0)
    semesters_completed = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.student.full_name} - CGPA: {self.cgpa}"
    
    def calculate_cgpa(self):
        """Calculate CGPA based on all semester SGPAs"""
        semester_results = SemesterResult.objects.filter(
            student=self.student,
            sgpa__isnull=False
        )
        
        total_grade_points = 0
        total_credits = 0
        
        for result in semester_results:
            total_grade_points += float(result.sgpa) * float(result.total_credits)
            total_credits += float(result.total_credits)
        
        if total_credits > 0:
            self.cgpa = total_grade_points / total_credits
            self.total_credits = total_credits
            self.semesters_completed = semester_results.count()
            self.save()
        
        return self.cgpa

class Admin(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    admin_id = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50, blank=True, null=True)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    role = models.CharField(max_length=50)  # Super Admin, Registrar, etc.
    permissions = models.JSONField(default=dict, blank=True)  # Custom permissions
    
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.role})"
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()
