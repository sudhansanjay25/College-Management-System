# users/urls.py
from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('password-change/', views.password_change_view, name='password-change'),
    
    # Departments
    path('departments/', views.DepartmentListCreateView.as_view(), name='department-list'),
    path('departments/<int:pk>/', views.DepartmentDetailView.as_view(), name='department-detail'),
    
    # Academic Years
    path('academic-years/', views.AcademicYearListCreateView.as_view(), name='academic-year-list'),
    
    # Students
    path('students/', views.StudentListCreateView.as_view(), name='student-list'),
    path('students/<int:pk>/', views.StudentDetailView.as_view(), name='student-detail'),
    
    # Teachers
    path('teachers/', views.TeacherListCreateView.as_view(), name='teacher-list'),
    path('teachers/<int:pk>/', views.TeacherDetailView.as_view(), name='teacher-detail'),
    
    # Subjects
    path('subjects/', views.SubjectListCreateView.as_view(), name='subject-list'),
    path('subjects/<int:pk>/', views.SubjectDetailView.as_view(), name='subject-detail'),
    
    # Student Marks
    path('marks/', views.StudentMarksListCreateView.as_view(), name='marks-list'),
    path('marks/<int:pk>/', views.StudentMarksDetailView.as_view(), name='marks-detail'),
    
    # Reports and Academic Summary
    path('student-report/', views.student_marks_report, name='student-report-own'),
    path('student-report/<str:student_id>/', views.student_marks_report, name='student-report'),
    path('academic-summary/', views.student_academic_summary, name='academic-summary-own'),
    path('academic-summary/<str:student_id>/', views.student_academic_summary, name='academic-summary'),
    
    # SGPA Calculation
    path('calculate-sgpa/<str:student_id>/<int:semester>/', views.calculate_semester_sgpa, name='calculate-sgpa'),
    
    # Face Recognition
    path('upload-face/', views.upload_face_encoding, name='upload-face'),
]