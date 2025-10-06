# users/views.py
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.shortcuts import get_object_or_404
from django.db import models
from .models import (User, Student, Teacher, Admin, Department, AcademicYear, 
                    Subject, ExamType, StudentMarks, SemesterResult, StudentCGPA)
from django.utils import timezone
from .serializers import *
import face_recognition
import numpy as np
from django.core.files.base import ContentFile
import base64

class DepartmentListCreateView(generics.ListCreateAPIView):
    queryset = Department.objects.filter(is_active=True)
    serializer_class = DepartmentSerializer
    permission_classes = [permissions.IsAuthenticated]

class DepartmentDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Department.objects.all()
    serializer_class = DepartmentSerializer
    permission_classes = [permissions.IsAuthenticated]

class AcademicYearListCreateView(generics.ListCreateAPIView):
    queryset = AcademicYear.objects.filter(is_active=True)
    serializer_class = AcademicYearSerializer
    permission_classes = [permissions.IsAuthenticated]

class StudentListCreateView(generics.ListCreateAPIView):
    queryset = Student.objects.filter(is_active=True)
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return StudentCreateSerializer
        return StudentSerializer
    
    def get_queryset(self):
        queryset = Student.objects.filter(is_active=True)
        department = self.request.query_params.get('department')
        semester = self.request.query_params.get('semester')
        section = self.request.query_params.get('section')
        
        if department:
            queryset = queryset.filter(department=department)
        if semester:
            queryset = queryset.filter(current_semester=semester)
        if section:
            queryset = queryset.filter(section=section)
            
        return queryset

class StudentDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def perform_destroy(self, instance):
        instance.is_active = False
        instance.save()

class TeacherListCreateView(generics.ListCreateAPIView):
    queryset = Teacher.objects.filter(is_active=True)
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return TeacherCreateSerializer
        return TeacherSerializer

class TeacherDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Teacher.objects.all()
    serializer_class = TeacherSerializer
    permission_classes = [permissions.IsAuthenticated]

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def upload_face_encoding(request):
    user_type = request.data.get('user_type')
    user_id = request.data.get('user_id')
    face_image = request.data.get('face_image')  # Base64 encoded image
    
    if not all([user_type, user_id, face_image]):
        return Response({'error': 'Missing required fields'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(face_image.split(',')[1])
        
        # Load image and generate face encoding
        image = face_recognition.load_image_file(ContentFile(image_data))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return Response({'error': 'No face found in image'}, status=status.HTTP_400_BAD_REQUEST)
        
        face_encoding = face_encodings[0]
        
        # Save encoding based on user type
        if user_type == 'student':
            student = get_object_or_404(Student, student_id=user_id)
            student.face_encoding = face_encoding.tobytes()
            # No longer track face enrollment date/time
            student.save(update_fields=['face_encoding'])
        elif user_type == 'teacher':
            teacher = get_object_or_404(Teacher, teacher_id=user_id)
            teacher.face_encoding = face_encoding.tobytes()
            # No longer track face enrollment date/time
            teacher.save(update_fields=['face_encoding'])
        
        return Response({'message': 'Face encoding saved successfully'}, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def login_view(request):
    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)
        
        # Get user profile data
        profile_data = None
        if user.user_type == 'student':
            try:
                student = Student.objects.get(user=user)
                profile_data = StudentSerializer(student).data
            except Student.DoesNotExist:
                pass
        elif user.user_type == 'teacher':
            try:
                teacher = Teacher.objects.get(user=user)
                profile_data = TeacherSerializer(teacher).data
            except Teacher.DoesNotExist:
                pass
        elif user.user_type == 'admin':
            try:
                admin = Admin.objects.get(user=user)
                profile_data = AdminSerializer(admin).data
            except Admin.DoesNotExist:
                pass
        
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'id': user.id,
                'username': user.username,
                'user_type': user.user_type,
                'must_change_password': getattr(user, 'must_change_password', False),
                'profile': profile_data
            }
        }, status=status.HTTP_200_OK)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def logout_view(request):
    try:
        refresh_token = request.data.get('refresh')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        return Response({'message': 'Successfully logged out'}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)  


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def password_change_view(request):
    serializer = PasswordChangeSerializer(data=request.data, context={'request': request})
    if serializer.is_valid():
        serializer.save()
        # Optionally issue new tokens after password change (forces refresh on clients)
        refresh = RefreshToken.for_user(request.user)
        return Response({
            'message': 'Password updated successfully',
            'refresh': str(refresh),
            'access': str(refresh.access_token)
        }, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Subject Views
class SubjectListCreateView(generics.ListCreateAPIView):
    queryset = Subject.objects.filter(is_active=True)
    serializer_class = SubjectSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = Subject.objects.filter(is_active=True)
        department = self.request.query_params.get('department')
        semester = self.request.query_params.get('semester')
        
        if department:
            queryset = queryset.filter(department=department)
        if semester:
            queryset = queryset.filter(semester=semester)
            
        return queryset

class SubjectDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer
    permission_classes = [permissions.IsAuthenticated]

# Student Marks Views
class StudentMarksListCreateView(generics.ListCreateAPIView):
    queryset = StudentMarks.objects.all()
    serializer_class = StudentMarksSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = StudentMarks.objects.all().select_related(
            'student', 'subject', 'exam_type', 'academic_year', 'entered_by'
        )
        
        # Filter based on user type
        user = self.request.user
        if hasattr(user, 'user_type'):
            if user.user_type == 'student':
                # Students can only see their own marks
                try:
                    student = Student.objects.get(user=user)
                    queryset = queryset.filter(student=student)
                except Student.DoesNotExist:
                    queryset = queryset.none()
            elif user.user_type == 'teacher':
                # Teachers can see marks they entered or from their department
                try:
                    teacher = Teacher.objects.get(user=user)
                    queryset = queryset.filter(
                        models.Q(entered_by=teacher) | 
                        models.Q(student__department=teacher.department)
                    )
                except Teacher.DoesNotExist:
                    queryset = queryset.none()
        
        # Additional filters
        student_id = self.request.query_params.get('student_id')
        subject_id = self.request.query_params.get('subject_id')
        exam_type = self.request.query_params.get('exam_type')
        semester = self.request.query_params.get('semester')
        
        if student_id:
            queryset = queryset.filter(student__student_id=student_id)
        if subject_id:
            queryset = queryset.filter(subject_id=subject_id)
        if exam_type:
            queryset = queryset.filter(exam_type__name=exam_type)
        if semester:
            queryset = queryset.filter(subject__semester=semester)
            
        return queryset.order_by('-entered_date')
    
    def perform_create(self, serializer):
        # Only teachers and admins can create marks
        if hasattr(self.request.user, 'user_type'):
            if self.request.user.user_type not in ['teacher', 'admin']:
                raise permissions.PermissionDenied("Only teachers and admins can enter marks")

class StudentMarksDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = StudentMarks.objects.all()
    serializer_class = StudentMarksSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_object(self):
        obj = super().get_object()
        user = self.request.user
        
        # Check permissions
        if hasattr(user, 'user_type'):
            if user.user_type == 'student':
                # Students can only view their own marks
                try:
                    student = Student.objects.get(user=user)
                    if obj.student != student:
                        raise permissions.PermissionDenied("You can only view your own marks")
                except Student.DoesNotExist:
                    raise permissions.PermissionDenied("Student profile not found")
            elif user.user_type == 'teacher':
                # Teachers can view/edit marks from their department
                try:
                    teacher = Teacher.objects.get(user=user)
                    if obj.student.department != teacher.department:
                        raise permissions.PermissionDenied("You can only access marks from your department")
                except Teacher.DoesNotExist:
                    raise permissions.PermissionDenied("Teacher profile not found")
        
        return obj

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def student_marks_report(request, student_id=None):
    """Get complete marks report for a student"""
    user = request.user
    
    # Determine which student's report to fetch
    if student_id:
        # If student_id provided, check permissions
        if hasattr(user, 'user_type'):
            if user.user_type == 'student':
                # Students can only view their own report
                try:
                    student = Student.objects.get(user=user)
                    if student.student_id != student_id:
                        return Response({'error': 'You can only view your own marks'}, 
                                      status=status.HTTP_403_FORBIDDEN)
                except Student.DoesNotExist:
                    return Response({'error': 'Student profile not found'}, 
                                  status=status.HTTP_404_NOT_FOUND)
            elif user.user_type == 'teacher':
                # Teachers can view reports from their department
                try:
                    teacher = Teacher.objects.get(user=user)
                    student = Student.objects.get(student_id=student_id)
                    if student.department != teacher.department:
                        return Response({'error': 'You can only view reports from your department'}, 
                                      status=status.HTTP_403_FORBIDDEN)
                except (Teacher.DoesNotExist, Student.DoesNotExist):
                    return Response({'error': 'Profile not found'}, 
                                  status=status.HTTP_404_NOT_FOUND)
        
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            return Response({'error': 'Student not found'}, status=status.HTTP_404_NOT_FOUND)
    else:
        # No student_id provided, use current user (only for students)
        if hasattr(user, 'user_type') and user.user_type == 'student':
            try:
                student = Student.objects.get(user=user)
            except Student.DoesNotExist:
                return Response({'error': 'Student profile not found'}, 
                              status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({'error': 'Student ID required'}, 
                          status=status.HTTP_400_BAD_REQUEST)
    
    # Get student's complete report
    serializer = StudentMarksReportSerializer(student)
    return Response(serializer.data, status=status.HTTP_200_OK)

# SGPA/CGPA calculation views
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def calculate_semester_sgpa(request, student_id, semester):
    """Calculate SGPA for a specific semester"""
    # Only teachers and admins can trigger SGPA calculation
    if hasattr(request.user, 'user_type'):
        if request.user.user_type not in ['teacher', 'admin']:
            return Response({'error': 'Only teachers and admins can calculate SGPA'}, 
                          status=status.HTTP_403_FORBIDDEN)
    
    try:
        student = Student.objects.get(student_id=student_id)
        academic_year = AcademicYear.objects.get(is_current=True)
        
        # Get or create semester result
        semester_result, created = SemesterResult.objects.get_or_create(
            student=student,
            semester=semester,
            academic_year=academic_year
        )
        
        # Calculate SGPA
        sgpa = semester_result.calculate_sgpa()
        
        if sgpa:
            # Update or create CGPA record
            cgpa_record, created = StudentCGPA.objects.get_or_create(student=student)
            cgpa_record.calculate_cgpa()
            
            return Response({
                'message': 'SGPA calculated successfully',
                'sgpa': sgpa,
                'cgpa': cgpa_record.cgpa
            }, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Unable to calculate SGPA. Check if semester marks are entered.'}, 
                          status=status.HTTP_400_BAD_REQUEST)
            
    except Student.DoesNotExist:
        return Response({'error': 'Student not found'}, status=status.HTTP_404_NOT_FOUND)
    except AcademicYear.DoesNotExist:
        return Response({'error': 'No current academic year found'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def student_academic_summary(request, student_id=None):
    """Get academic summary including SGPA and CGPA"""
    user = request.user
    
    # Determine which student's summary to fetch (similar logic as marks report)
    if student_id:
        if hasattr(user, 'user_type'):
            if user.user_type == 'student':
                try:
                    student = Student.objects.get(user=user)
                    if student.student_id != student_id:
                        return Response({'error': 'You can only view your own academic summary'}, 
                                      status=status.HTTP_403_FORBIDDEN)
                except Student.DoesNotExist:
                    return Response({'error': 'Student profile not found'}, 
                                  status=status.HTTP_404_NOT_FOUND)
        
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            return Response({'error': 'Student not found'}, status=status.HTTP_404_NOT_FOUND)
    else:
        if hasattr(user, 'user_type') and user.user_type == 'student':
            try:
                student = Student.objects.get(user=user)
            except Student.DoesNotExist:
                return Response({'error': 'Student profile not found'}, 
                              status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({'error': 'Student ID required'}, 
                          status=status.HTTP_400_BAD_REQUEST)
    
    # Get semester results and CGPA
    semester_results = SemesterResult.objects.filter(student=student).order_by('semester')
    try:
        cgpa_record = StudentCGPA.objects.get(student=student)
    except StudentCGPA.DoesNotExist:
        cgpa_record = None
    
    data = {
        'student': StudentSerializer(student).data,
        'semester_results': SemesterResultSerializer(semester_results, many=True).data,
        'cgpa_record': StudentCGPASerializer(cgpa_record).data if cgpa_record else None
    }
    
    return Response(data, status=status.HTTP_200_OK)