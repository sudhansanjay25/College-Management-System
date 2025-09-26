# Attendance System Backend – Role-Based Login & ID Generation

## Goal
Provide separate login capability for students, teachers, and admins:
- Admins: Django superuser / staff (Django Admin UI)
- Students & Teachers: Login via API using auto-generated IDs + default passwords
- Each Student/Teacher gets an auto-created `User` record (custom user model with `user_type`)

## User Types
`User.user_type` (choices):
- `admin`
- `teacher`
- `student`

Students/Teachers are NOT staff or superusers; they use frontend/UI, not Django Admin (except for creation by admin staff).

## Unique ID Generation
Pattern: `<YY><DEPTCODE><NNN>`
- YY: 2-digit year (from `AcademicYear.start_date.year % 100`, fallback: first segment of `year` field)
- DEPTCODE: `Department.code` (uppercase) else first 3 letters of name
- NNN: 3-digit serial (either provided roll number or auto sequence per (year, dept) cluster)

Examples:
- AcademicYear 2024-2025, Department code CS, first student → `24CS001`
- Next student same year/department → `24CS002`
- Teacher IDs follow same concept (if teachers do not link to `AcademicYear`, YY may degrade; adjust if model adds it later)

Collision handling: If generated ID exists, serial is incremented until unique.

## Automatic User Creation (Admin Save Hooks)
On creating Student/Teacher in Django Admin:
1. If `student_id` / `teacher_id` is empty → generate it
2. Create `User` with:
   - `username = generated_id`
   - `user_type = student|teacher`
   - `password = student123` or `teacher123`
3. Link via OneToOne (`user` field on profile)
4. Keep `User.username` synced if ID regenerates (rare; IDs set read-only to prevent manual edits)

## Default Passwords
- Student: `student123`
- Teacher: `teacher123`
Recommendation: Force password change on first login (implement endpoint later).

## Admin Workflow
1. Create Department(s) with a short code (required for nicer IDs)
2. Create AcademicYear (set `start_date`, `end_date`, `is_current`)
3. Add Student:
   - Leave ID read-only (auto)
   - Provide names, email, department, academic year, optional roll number
4. Add Teacher similarly (with designation, department)
5. Use generated ID + default password at login endpoint

## Login API (Example)
POST /api/users/login/
```
{
  "username": "24CS001",
  "password": "student123"
}
```
Response should include tokens + user profile (depends on your serializer implementation).

If you need serializer/view patch examples, add those later.

## Windows Folder Creation (Management Command Example)
If you later add a management command for test users:
```
mkdir users\management
mkdir users\management\commands
type NUL > users\management\__init__.py
type NUL > users\management\commands\__init__.py
```
Then add `create_test_users.py` inside `commands/`.

## Suggested Management Command (Optional)
Creates demo dept/year/student/teacher for quick testing. (Not included here—ask if needed.)

## Password Change Flow (Future)
Implement endpoint:
- Auth required
- Accept old_password, new_password
- Validate & call `set_password`
- Invalidate tokens if using JWT (rotate refresh if needed)

## Security Notes
- Enforce password change after first login (store `must_change_password` boolean)
- Never expose default passwords in production logs
- Consider rate limiting login (already have `django-ratelimit`)
- Ensure `is_active` flag respected in authentication

## Extensibility
Future ideas:
- Add soft-delete flags
- Track last login, password updated at
- Add audit logging (signals or middleware)
- Attach permissions per `user_type`

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Login fails | No linked User | Ensure admin save hook executed; recreate profile |
| ID not generated | Missing academic year or department | Provide both; check admin override loaded |
| Wrong year code | Missing `start_date` | Populate `AcademicYear.start_date` |
| Duplicate IDs | Manual edits (should not happen) | Keep field read-only |

## Minimal Developer Checklist
- [ ] Migrate DB (models with `user_type`)
- [ ] Create Departments + AcademicYear
- [ ] Add Student/Teacher via admin
- [ ] Test login with generated ID
- [ ] Add password change endpoint
- [ ] Implement role-based routing in frontend

## Example Response Structure (Desired)
```
{
  "access": "...",
  "refresh": "...",
  "user": {
    "id": 12,
    "username": "24CS001",
    "user_type": "student",
    "profile": {
       "student_id": "24CS001",
       "full_name": "John Doe",
       "department": "Computer Science",
       "current_semester": 1
    }
  }
}
```

## Summary
Core principle: decouple authentication (User) from academic/profile data (Student/Teacher), but enforce 1:1 linkage and consistent, generated IDs used as usernames.

Ask if you need:
- Serializer/view patches
- Management command code
- Password change endpoint
- Frontend routing guidance
