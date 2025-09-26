# Academic Management System Updates

This document outlines the major updates made to the Attendance System to support comprehensive academic management including marks, SGPA, and CGPA calculations.

## Changes Made

### 1. Admin Superuser Creation
- **Change**: When a new admin is created through Django admin, they are automatically made a Django superuser
- **Implementation**: Modified `AdminModelAdmin.save_model()` to set `is_staff=True` and `is_superuser=True`

### 2. Teacher Permissions
- **Change**: Teachers now have authority to create, modify, and delete student data
- **Implementation**: 
  - Teachers are given `is_staff=True` to access Django admin
  - Added permission checks in `StudentAdmin`, `StudentMarksAdmin`
  - Teachers can manage students from their department

### 3. Subject Management with Credits
- **New Model**: `Subject`
  - Links to Department and Semester
  - Includes credits for SGPA calculation
  - Has practical exam flag
  - Unique per department-semester-code combination

### 4. Comprehensive Exam System
- **New Model**: `ExamType` with predefined types:
  - Internal Exam 1 (15% weightage)
  - Internal Exam 2 (15% weightage)
  - Practical Exam (20% weightage)
  - Semester End Exam (50% weightage)

- **New Model**: `StudentMarks`
  - Stores marks for each student-subject-exam combination
  - Tracks who entered the marks (teacher)
  - Calculates percentage automatically

### 5. SGPA/CGPA Calculation System
- **New Model**: `SemesterResult`
  - Calculates SGPA based on semester end exam marks
  - Uses 10-point grading scale
  - Weighted by subject credits

- **New Model**: `StudentCGPA`
  - Calculates overall CGPA from all semester SGPAs
  - Tracks total credits and semesters completed

## New API Endpoints

### Subjects
- `GET/POST /api/users/subjects/` - List/Create subjects
- `GET/PUT/DELETE /api/users/subjects/{id}/` - Subject details

### Student Marks
- `GET/POST /api/users/marks/` - List/Create marks
- `GET/PUT/DELETE /api/users/marks/{id}/` - Mark details

### Reports
- `GET /api/users/student-report/` - Current student's complete marks report
- `GET /api/users/student-report/{student_id}/` - Specific student's marks report
- `GET /api/users/academic-summary/` - Current student's SGPA/CGPA summary
- `GET /api/users/academic-summary/{student_id}/` - Specific student's summary

### SGPA Calculation
- `POST /api/users/calculate-sgpa/{student_id}/{semester}/` - Calculate SGPA for semester

## Permission System

### Students
- Can view only their own marks and academic reports
- Cannot modify any marks or academic data

### Teachers
- Can view/modify marks for students in their department
- Can create, modify, and delete student records
- Can calculate SGPA for students
- Have access to Django admin interface

### Admins
- Full superuser permissions
- Can manage all data across all departments

## Database Schema

### New Tables
1. `users_subject` - Subject information with credits
2. `users_examtype` - Exam type definitions
3. `users_studentmarks` - Student marks for each exam
4. `users_semesterresult` - Semester SGPA records
5. `users_studentcgpa` - Overall CGPA records

### Key Relationships
- Subject → Department (Many-to-One)
- StudentMarks → Student, Subject, ExamType (Many-to-One)
- SemesterResult → Student, AcademicYear (Many-to-One)
- StudentCGPA → Student (One-to-One)

## Grading Scale (10-point system)
- 90-100%: Grade Point 10
- 80-89%: Grade Point 9
- 70-79%: Grade Point 8
- 60-69%: Grade Point 7
- 50-59%: Grade Point 6
- 40-49%: Grade Point 5
- Below 40%: Grade Point 0

## Migration Files Created
1. `0003_add_academic_models.py` - Creates all new academic models

## Management Commands
1. `create_exam_types.py` - Populates default exam types with weightages

## Implementation Notes

### SGPA Calculation Logic
```python
SGPA = Σ(Grade Point × Credits) / Σ(Credits)
```

### CGPA Calculation Logic  
```python
CGPA = Σ(SGPA × Semester Credits) / Σ(All Credits)
```

### Security Features
- Role-based access control at view level
- Teachers can only access their department's data
- Students can only access their own data
- Audit trail with entered_by tracking

## Usage Examples

### For Teachers - Entering Marks
1. Login with teacher credentials
2. Navigate to Django admin → Student Marks
3. Add new mark entry selecting student, subject, exam type
4. System automatically tracks who entered the marks

### For Students - Viewing Reports
1. Login with student credentials
2. GET `/api/users/student-report/` to see all marks
3. GET `/api/users/academic-summary/` to see SGPA/CGPA

### Calculating SGPA
1. Ensure all semester end exam marks are entered
2. POST to `/api/users/calculate-sgpa/{student_id}/{semester}/`
3. System calculates SGPA and updates CGPA automatically

## Error Handling
- Permission denied for unauthorized access
- Validation for mark ranges (0 to max_marks)
- Unique constraints prevent duplicate mark entries
- Graceful handling of missing student/teacher profiles