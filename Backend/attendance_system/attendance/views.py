# attendance/views.py
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db.models import Count, Q, Avg
from django.utils import timezone
from datetime import datetime, timedelta
from .models import Subject, SubjectTeacher, SubjectEnrollment, ClassSchedule, AttendanceSession, Attendance
from .serializers import *
from users.models import Student, Teacher
import face_recognition
import numpy as np

class SubjectListCreateView(generics.ListCreateAPIView):
    queryset = Subject.objects.filter(is_active=True)
    serializer_class = SubjectSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = Subject.objects.filter(is_active=True)
        department = self.request.query_params.get('department')
        semester = self.request.query_params.get('semester')
        teacher = self.request.query_params.get('teacher')
        
        if department:
            queryset = queryset.filter(department=department)
        if semester:
            queryset = queryset.filter(semester=semester)
        if teacher:
            queryset = queryset.filter(teachers=teacher)
            
        return queryset

class SubjectDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer
    permission_classes = [permissions.IsAuthenticated]

class ClassScheduleListCreateView(generics.ListCreateAPIView):
    queryset = ClassSchedule.objects.filter(is_active=True)
    serializer_class = ClassScheduleSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = ClassSchedule.objects.filter(is_active=True)
        teacher = self.request.query_params.get('teacher')
        department = self.request.query_params.get('department')
        day = self.request.query_params.get('day')
        
        if teacher:
            queryset = queryset.filter(teacher=teacher)
        if department:
            queryset = queryset.filter(subject__department=department)
        if day:
            queryset = queryset.filter(day_of_week=day)
            
        return queryset.order_by('day_of_week', 'start_time')

class AttendanceSessionListCreateView(generics.ListCreateAPIView):
    queryset = AttendanceSession.objects.all()
    serializer_class = AttendanceSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = AttendanceSession.objects.all()
        subject = self.request.query_params.get('subject')
        teacher = self.request.query_params.get('teacher')
        date = self.request.query_params.get('date')
        
        if subject:
            queryset = queryset.filter(subject=subject)
        if teacher:
            queryset = queryset.filter(teacher=teacher)
        if date:
            queryset = queryset.filter(date=date)
            
        return queryset.order_by('-date', '-start_time')

class AttendanceSessionDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = AttendanceSession.objects.all()
    serializer_class = AttendanceSessionSerializer
    permission_classes = [permissions.IsAuthenticated]

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def start_attendance_session(request):
    """Start a new attendance session"""
    subject_id = request.data.get('subject_id')
    teacher_id = request.data.get('teacher_id')
    session_type = request.data.get('session_type', 'regular')
    
    try:
        subject = Subject.objects.get(id=subject_id)
        teacher = Teacher.objects.get(id=teacher_id)
        
        # Check if session already exists for today
        today = timezone.now().date()
        existing_session = AttendanceSession.objects.filter(
            subject=subject,
            teacher=teacher,
            date=today,
            is_completed=False
        ).first()
        
        if existing_session:
            return Response({
                'message': 'Session already active',
                'session': AttendanceSessionSerializer(existing_session).data
            }, status=status.HTTP_200_OK)
        
        # Get enrolled students count
        enrolled_count = SubjectEnrollment.objects.filter(
            subject=subject, 
            is_active=True
        ).count()
        
        # Create new session
        session = AttendanceSession.objects.create(
            subject=subject,
            teacher=teacher,
            date=today,
            start_time=timezone.now().time(),
            session_type=session_type,
            total_enrolled=enrolled_count,
            is_active=True
        )
        
        return Response({
            'message': 'Attendance session started successfully',
            'session': AttendanceSessionSerializer(session).data
        }, status=status.HTTP_201_CREATED)
        
    except (Subject.DoesNotExist, Teacher.DoesNotExist) as e:
        return Response({'error': 'Subject or Teacher not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def mark_attendance_face(request):
    """Mark attendance using face recognition"""
    session_id = request.data.get('session_id')
    face_image = request.data.get('face_image')  # Base64 encoded
    device_id = request.data.get('device_id')
    
    try:
        session = get_object_or_404(AttendanceSession, id=session_id, is_active=True)
        
        # Decode and process face image
        import base64
        from django.core.files.base import ContentFile
        
        image_data = base64.b64decode(face_image.split(',')[1])
        image = face_recognition.load_image_file(ContentFile(image_data))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)
        
        unknown_face_encoding = face_encodings[0]
        
        # Get all enrolled students for this subject
        enrolled_students = Student.objects.filter(
            subjectenrollment__subject=session.subject,
            subjectenrollment__is_active=True,
            face_encoding__isnull=False
        )
        
        best_match = None
        best_distance = float('inf')
        
        for student in enrolled_students:
            # Check if attendance already marked
            existing_attendance = Attendance.objects.filter(
                session=session,
                student=student
            ).first()
            
            if existing_attendance:
                continue
                
            # Compare face encodings
            stored_encoding = np.frombuffer(student.face_encoding, dtype=np.float64)
            distance = face_recognition.face_distance([stored_encoding], unknown_face_encoding)[0]
            
            if distance < best_distance and distance < 0.6:  # Threshold for face match
                best_distance = distance
                best_match = student
        
        if best_match:
            # Determine status based on time
            current_time = timezone.now().time()
            grace_period = timedelta(minutes=10)
            session_start_datetime = datetime.combine(session.date, session.start_time)
            late_threshold = (session_start_datetime + grace_period).time()
            
            status_value = 'late' if current_time > late_threshold else 'present'
            
            # Create attendance record
            attendance = Attendance.objects.create(
                session=session,
                student=best_match,
                status=status_value,
                face_match_confidence=round(1 - best_distance, 4),
                device_used_id=device_id,
                is_manual_entry=False
            )
            
            # Update session counters
            if status_value == 'present':
                session.total_present += 1
            elif status_value == 'late':
                session.total_late += 1
            session.save()
            
            return Response({
                'message': f'Attendance marked as {status_value}',
                'student': {
                    'name': best_match.full_name,
                    'student_id': best_match.student_id
                },
                'status': status_value,
                'confidence': round(1 - best_distance, 4)
            }, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Face not recognized'}, status=status.HTTP_404_NOT_FOUND)
            
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def complete_attendance_session(request, session_id):
    """Complete an attendance session and mark remaining students as absent"""
    try:
        session = get_object_or_404(AttendanceSession, id=session_id)
        
        if session.is_completed:
            return Response({'error': 'Session already completed'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Mark remaining enrolled students as absent
        enrolled_students = Student.objects.filter(
            subjectenrollment__subject=session.subject,
            subjectenrollment__is_active=True
        )
        
        for student in enrolled_students:
            attendance, created = Attendance.objects.get_or_create(
                session=session,
                student=student,
                defaults={
                    'status': 'absent',
                    'is_manual_entry': True
                }
            )
            
            if created and attendance.status == 'absent':
                session.total_absent += 1
        
        # Complete the session
        session.is_completed = True
        session.completed_at = timezone.now()
        session.end_time = timezone.now().time()
        session.save()
        
        return Response({
            'message': 'Attendance session completed successfully',
            'session': AttendanceSessionSerializer(session).data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def attendance_report(request):
    """Generate attendance report for students"""
    subject_id = request.query_params.get('subject_id')
    student_id = request.query_params.get('student_id')
    start_date = request.query_params.get('start_date')
    end_date = request.query_params.get('end_date')
    
    try:
        queryset = Attendance.objects.all()
        
        if subject_id:
            queryset = queryset.filter(session__subject_id=subject_id)
        if student_id:
            queryset = queryset.filter(student_id=student_id)
        if start_date:
            queryset = queryset.filter(session__date__gte=start_date)
        if end_date:
            queryset = queryset.filter(session__date__lte=end_date)
        
        # Group by student and calculate statistics
        from django.db.models import Count, Case, When, IntegerField
        
        report_data = []
        students = Student.objects.filter(id__in=queryset.values_list('student', flat=True).distinct())
        
        for student in students:
            student_attendance = queryset.filter(student=student)
            
            total_classes = student_attendance.count()
            present_count = student_attendance.filter(status='present').count()
            absent_count = student_attendance.filter(status='absent').count()
            late_count = student_attendance.filter(status='late').count()
            
            attendance_percentage = (present_count + late_count) / total_classes * 100 if total_classes > 0 else 0
            
            report_data.append({
                'student': StudentSerializer(student).data,
                'total_classes': total_classes,
                'present_count': present_count,
                'absent_count': absent_count,
                'late_count': late_count,
                'attendance_percentage': round(attendance_percentage, 2)
            })
        
        return Response(report_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
