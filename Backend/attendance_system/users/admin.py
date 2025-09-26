# users/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib import messages
from django import forms
from .models import (User, Student, Teacher, Admin, Department, AcademicYear, 
                    Subject, ExamType, StudentMarks, SemesterResult, StudentCGPA)
from django.db import models  # for isinstance checks if needed

@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'is_active']
    list_filter = ['is_active', 'established_date']
    search_fields = ['name', 'code', 'description']

@admin.register(AcademicYear)
class AcademicYearAdmin(admin.ModelAdmin):
    list_display = ['year', 'start_date', 'end_date', 'is_current', 'is_active']
    list_filter = ['is_current', 'is_active']
    search_fields = ['year']

# @admin.register(Student)
# class StudentAdmin(admin.ModelAdmin):
#     list_display = ['student_id', 'full_name', 'department', 'current_semester', 'section', 'is_active']
#     list_filter = ['department', 'current_semester', 'section', 'is_active', 'academic_year']
#     search_fields = ['student_id', 'first_name', 'last_name', 'email']
#     readonly_fields = ['enrollment_date', 'face_enrollment_date']

# @admin.register(Teacher)
# class TeacherAdmin(admin.ModelAdmin):
#     list_display = ['teacher_id', 'full_name', 'department', 'designation', 'is_active']
#     list_filter = ['department', 'designation', 'is_active']
#     search_fields = ['teacher_id', 'first_name', 'last_name', 'email']
#     readonly_fields = ['joining_date', 'face_enrollment_date']

#Changes for login
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'user_type', 'first_name', 'last_name', 'is_active']
    list_filter = ['user_type', 'is_active']
    
    # Add user_type to the fieldsets
    fieldsets = UserAdmin.fieldsets + (
        ('User Type', {'fields': ('user_type',)}),
    )

# Helpers for ID generation
def _year_code(academic_year) -> str:
    # Try start_date first
    try:
        return str(academic_year.start_date.year % 100).zfill(2)
    except Exception:
        pass
    # Then try parsing "YYYY-YYYY" or "YYYY" from academic_year.year
    try:
        y = str(academic_year.year)
        if "-" in y:
            return y.split("-")[0][-2:]
        return y[-2:]
    except Exception:
        return "00"

def _dept_code(dept) -> str:
    try:
        if getattr(dept, "code", None):
            return str(dept.code).upper()
        return str(dept.name)[:3].upper()
    except Exception:
        return "GEN"

def _normalize_serial(roll_number, fallback_count: int) -> str:
    # Use provided roll number if given; otherwise use fallback sequence
    try:
        if roll_number is not None and str(roll_number).strip() != "":
            return str(int(roll_number)).zfill(3)
    except Exception:
        pass
    return str(int(fallback_count)).zfill(3)

def _generate_unique_id(model_cls, id_field_name: str, academic_year, department, roll_number=None) -> str:
    year_code = _year_code(academic_year)
    dept_code = _dept_code(department)

    # Fallback sequence: count existing in same (year, dept) + 1
    base_qs = model_cls.objects.all()
    if hasattr(model_cls, "academic_year"):
        base_qs = base_qs.filter(academic_year=academic_year)
    if hasattr(model_cls, "department"):
        base_qs = base_qs.filter(department=department)
    serial = _normalize_serial(roll_number, base_qs.count() + 1)

    candidate = f"{year_code}{dept_code}{serial}"
    bump = 0
    while model_cls.objects.filter(**{id_field_name: candidate}).exists():
        bump += 1
        candidate = f"{year_code}{dept_code}{str(int(serial) + bump).zfill(3)}"
    return candidate

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ['student_id', 'full_name', 'department', 'current_semester', 'section', 'is_active']
    list_filter = ['department', 'current_semester', 'section', 'academic_year', 'is_active']
    search_fields = ['student_id', 'first_name', 'last_name', 'email']
    autocomplete_fields = ['department', 'academic_year']
    readonly_fields = ['student_id']  # keep auto-generated id immutable
    
    def has_add_permission(self, request):
        # Teachers and admins can add students
        if hasattr(request.user, 'user_type'):
            return request.user.user_type in ['teacher', 'admin']
        return request.user.is_superuser
    
    def has_change_permission(self, request, obj=None):
        # Teachers and admins can modify students
        if hasattr(request.user, 'user_type'):
            return request.user.user_type in ['teacher', 'admin']
        return request.user.is_superuser
    
    def has_delete_permission(self, request, obj=None):
        # Teachers and admins can delete students
        if hasattr(request.user, 'user_type'):
            return request.user.user_type in ['teacher', 'admin']
        return request.user.is_superuser

    def save_model(self, request, obj, form, change):
        # Ensure student_id exists or (re)generate if missing
        if not getattr(obj, 'student_id', None):
            roll = getattr(obj, 'roll_number', None) or getattr(obj, 'rollno', None) or getattr(obj, 'roll', None)
            obj.student_id = _generate_unique_id(Student, "student_id", obj.academic_year, obj.department, roll)

        # Auto-create linked User for new records (or if user not linked)
        if not change or not getattr(obj, 'user_id', None):
            if not getattr(obj, 'user', None):
                user = User.objects.create_user(
                    username=obj.student_id,
                    email=getattr(obj, 'email', '') or '',
                    first_name=getattr(obj, 'first_name', '') or '',
                    last_name=getattr(obj, 'last_name', '') or '',
                    user_type='student',
                    password='student123'
                )
                user.must_change_password = True
                user.save(update_fields=['must_change_password'])
                obj.user = user

        super().save_model(request, obj, form, change)

        # Keep User.username in sync with the generated student_id
        if getattr(obj, 'user', None) and obj.user.username != obj.student_id:
            obj.user.username = obj.student_id
            obj.user.save(update_fields=['username'])


@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    list_display = ['teacher_id', 'full_name', 'department', 'designation', 'is_active']
    list_filter = ['department', 'designation', 'is_active']
    search_fields = ['teacher_id', 'first_name', 'last_name', 'email']
    readonly_fields = ['teacher_id']  # keep auto-generated id immutable

    def save_model(self, request, obj, form, change):
        # Ensure teacher_id exists or (re)generate if missing
        if not getattr(obj, 'teacher_id', None):
            roll = getattr(obj, 'roll_number', None) or getattr(obj, 'rollno', None) or getattr(obj, 'roll', None)
            obj.teacher_id = _generate_unique_id(Teacher, "teacher_id", obj.academic_year if hasattr(obj, 'academic_year') else None, obj.department, roll)

        # Auto-create linked User for new records (or if user not linked)
        if not change or not getattr(obj, 'user_id', None):
            if not getattr(obj, 'user', None):
                user = User.objects.create_user(
                    username=obj.teacher_id,
                    email=getattr(obj, 'email', '') or '',
                    first_name=getattr(obj, 'first_name', '') or '',
                    last_name=getattr(obj, 'last_name', '') or '',
                    user_type='teacher',
                    password='teacher123',
                    is_staff=True  # Give teachers staff permissions to access admin
                )
                user.must_change_password = True
                user.save(update_fields=['must_change_password'])
                obj.user = user

        super().save_model(request, obj, form, change)

        # Keep User.username in sync with the generated teacher_id
        if getattr(obj, 'user', None) and obj.user.username != obj.teacher_id:
            obj.user.username = obj.teacher_id
            obj.user.save(update_fields=['username'])

class AdminForm(forms.ModelForm):
    """Form used when adding / editing Admin objects.

    We EXCLUDE the underlying `user` field so the admin cannot accidentally
    pick a student / teacher user. A new privileged User will always be
    created automatically when adding an Admin. On edit, we only show the
    generated identifiers / profile fields (handled via ModelAdmin)."""

    class Meta:
        model = Admin
        exclude = ['user']  # user handled automatically

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only touch admin_id if it is present in form fields
        if 'admin_id' in self.fields:
            if not self.instance.pk:  # add form
                self.fields['admin_id'].widget = forms.HiddenInput()
                self.fields['admin_id'].required = False
            else:  # edit form
                self.fields['admin_id'].disabled = True

def _next_admin_id():
    """Return a new admin id (ADM###) unique across Admin.admin_id and User.username.

    We look at both existing Admin rows and any User rows whose username begins
    with 'ADM' to avoid collisions with orphaned user accounts (e.g. when an
    Admin object was deleted earlier without deleting its User record)."""
    existing_ids = set(
        Admin.objects.filter(admin_id__startswith='ADM').values_list('admin_id', flat=True)
    )
    existing_usernames = set(
        User.objects.filter(username__startswith='ADM').values_list('username', flat=True)
    )
    used = existing_ids | existing_usernames

    # Extract numeric suffixes
    nums = []
    for s in used:
        if len(s) > 3 and s[3:].isdigit():
            nums.append(int(s[3:]))
    current_max = max(nums) if nums else 0
    candidate_num = current_max + 1
    while True:
        cand = f"ADM{candidate_num:03d}"
        if cand not in used and not User.objects.filter(username=cand).exists():
            return cand
        candidate_num += 1

@admin.register(Admin)
class AdminModelAdmin(admin.ModelAdmin):
    form = AdminForm
    list_display = ['admin_id', 'full_name', 'role']
    search_fields = ['admin_id', 'first_name', 'last_name', 'email']

    def get_readonly_fields(self, request, obj=None):
        # admin_id becomes readonly after creation; also show linked username
        ro = []
        if obj:
            ro.append('admin_id')
        return ro

    def get_fields(self, request, obj=None):
        # Always include admin_id so the form can manage / hide it safely
        return ['admin_id', 'first_name', 'last_name', 'email', 'phone_number', 'role', 'permissions']

    def save_model(self, request, obj, form, change):
        # Ensure we have a unique admin_id (avoid existing username collisions)
        if not obj.admin_id:
            obj.admin_id = _next_admin_id()
        else:
            if User.objects.filter(username=obj.admin_id).exclude(pk=getattr(obj.user, 'pk', None)).exists():
                # If that username belongs to an orphan admin user (user_type='admin' without Admin record), reuse it
                orphan = User.objects.filter(
                    username=obj.admin_id,
                    user_type='admin',
                    admin__isnull=True
                ).first()
                if orphan:
                    obj.user = orphan
                else:
                    obj.admin_id = _next_admin_id()

        if not change:
            # If we already picked up an orphan user, reuse & elevate it
            if getattr(obj, 'user', None):
                u = obj.user
                updates = []
                # Guarantee privilege flags
                if not u.is_staff:
                    u.is_staff = True; updates.append('is_staff')
                if not u.is_superuser:
                    u.is_superuser = True; updates.append('is_superuser')
                # Sync profile
                for fld, val in [('first_name', obj.first_name), ('last_name', obj.last_name), ('email', obj.email)]:
                    if getattr(u, fld) != val:
                        setattr(u, fld, val); updates.append(fld)
                if updates:
                    u.save(update_fields=updates)
            else:
                user = User.objects.create_user(
                    username=obj.admin_id,
                    email=obj.email or '',
                    first_name=obj.first_name or '',
                    last_name=obj.last_name or '',
                    user_type='admin',
                    password='admin123',
                    is_staff=True,
                    is_superuser=True
                )
                user.must_change_password = True
                user.save(update_fields=['must_change_password'])
                obj.user = user
        else:
            # Keep linked user info consistent & privileged
            if obj.user:
                updates = []
                if obj.user.username != obj.admin_id:
                    # Guard against collision when editing
                    if User.objects.filter(username=obj.admin_id).exclude(pk=obj.user.pk).exists():
                        obj.admin_id = _next_admin_id()
                    obj.user.username = obj.admin_id
                    updates.append('username')
                for fld, val in [('first_name', obj.first_name), ('last_name', obj.last_name), ('email', obj.email)]:
                    if getattr(obj.user, fld) != val:
                        setattr(obj.user, fld, val)
                        updates.append(fld)
                # enforce privilege flags
                if not obj.user.is_staff:
                    obj.user.is_staff = True; updates.append('is_staff')
                if not obj.user.is_superuser:
                    obj.user.is_superuser = True; updates.append('is_superuser')
                if updates:
                    obj.user.save(update_fields=updates)

        super().save_model(request, obj, form, change)

        if not change and obj.user:
            messages.success(
                request,
                'Admin created successfully (existing user reused).' if User.objects.filter(username=obj.admin_id, admin__pk=obj.pk).count() == 1 else 'Admin created successfully!\n'
                f'Username: {obj.admin_id}\nPassword: admin123\n'
                '(Prompted to change password on first login)'
            )

    def delete_model(self, request, obj):
        # Remove the linked user so it does not appear elsewhere
        user = obj.user if obj and obj.user_id else None
        super().delete_model(request, obj)
        if user:
            user.delete()

    def delete_queryset(self, request, queryset):
        users = [a.user for a in queryset if a.user_id]
        super().delete_queryset(request, queryset)
        for u in users:
            u.delete()

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'department', 'semester', 'credits', 'has_practical', 'is_active']
    list_filter = ['department', 'semester', 'has_practical', 'is_active']
    search_fields = ['name', 'code', 'department__name']
    autocomplete_fields = ['department']
    list_editable = ['has_practical', 'is_active']

@admin.register(ExamType)
class ExamTypeAdmin(admin.ModelAdmin):
    list_display = ['name', 'display_name', 'max_marks', 'weightage_percentage']
    list_editable = ['max_marks', 'weightage_percentage']

@admin.register(StudentMarks)
class StudentMarksAdmin(admin.ModelAdmin):
    list_display = ['student', 'subject', 'exam_type', 'marks_obtained', 'max_marks', 'percentage', 'academic_year', 'entered_by']
    list_filter = ['exam_type', 'subject__department', 'subject__semester', 'academic_year']
    search_fields = ['student__student_id', 'student__first_name', 'student__last_name', 'subject__name']
    autocomplete_fields = ['student', 'subject', 'entered_by']
    
    def percentage(self, obj):
        return f"{obj.percentage:.2f}%"
    percentage.short_description = 'Percentage'
    
    def has_add_permission(self, request):
        # Only teachers and admins can add marks
        if hasattr(request.user, 'user_type'):
            return request.user.user_type in ['teacher', 'admin']
        return request.user.is_superuser
    
    def has_change_permission(self, request, obj=None):
        # Only teachers and admins can modify marks
        if hasattr(request.user, 'user_type'):
            return request.user.user_type in ['teacher', 'admin']
        return request.user.is_superuser
    
    def has_delete_permission(self, request, obj=None):
        # Only teachers and admins can delete marks
        if hasattr(request.user, 'user_type'):
            return request.user.user_type in ['teacher', 'admin']
        return request.user.is_superuser

@admin.register(SemesterResult)
class SemesterResultAdmin(admin.ModelAdmin):
    list_display = ['student', 'semester', 'academic_year', 'sgpa', 'total_credits', 'calculated_date']
    list_filter = ['semester', 'academic_year', 'student__department']
    search_fields = ['student__student_id', 'student__first_name', 'student__last_name']
    readonly_fields = ['sgpa', 'total_credits', 'calculated_date']
    actions = ['calculate_sgpa_for_selected']
    
    def calculate_sgpa_for_selected(self, request, queryset):
        for result in queryset:
            result.calculate_sgpa()
        self.message_user(request, f"SGPA calculated for {queryset.count()} semester results.")
    calculate_sgpa_for_selected.short_description = "Calculate SGPA for selected results"

@admin.register(StudentCGPA)
class StudentCGPAAdmin(admin.ModelAdmin):
    list_display = ['student', 'cgpa', 'total_credits', 'semesters_completed', 'last_updated']
    list_filter = ['student__department', 'semesters_completed']
    search_fields = ['student__student_id', 'student__first_name', 'student__last_name']
    readonly_fields = ['cgpa', 'total_credits', 'semesters_completed', 'last_updated']
    actions = ['calculate_cgpa_for_selected']
    
    def calculate_cgpa_for_selected(self, request, queryset):
        for cgpa_record in queryset:
            cgpa_record.calculate_cgpa()
        self.message_user(request, f"CGPA calculated for {queryset.count()} students.")
    calculate_cgpa_for_selected.short_description = "Calculate CGPA for selected students"
