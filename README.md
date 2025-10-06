# Centralized College Management System

A comprehensive, full-stack College Management System built with Django REST Framework backend and React.js frontend. This system provides a complete solution for managing students, teachers, academics, attendance, and devices with advanced face recognition technology and role-based access control.

## 🚀 Key Features

### **Authentication & User Management**
*   **Role-Based Authentication:** Secure JWT-based authentication for three user roles:
    *   **Admins:** Full system access with management capabilities
    *   **Teachers:** Department-specific permissions with student and academic data management
    *   **Students:** Access to personal academic records and attendance data
*   **Automatic User & ID Generation:** Auto-generated unique IDs for students, teachers, and admins
*   **Profile Management:** Complete profile management with photo uploads and face recognition enrollment

### **Academic Management System**
*   **Department & Year Management:** Centralized department management with academic year tracking
*   **Subject Management:** Comprehensive subject management with credits, semester mapping, and teacher assignments
*   **Exam & Marks Management:** 
    *   Multiple exam types (Internal 1, Internal 2, Practical, Semester End)
    *   Configurable weightages for different examination patterns
    *   Detailed mark entry and tracking system
*   **SGPA & CGPA Calculation:** 
    *   Automated SGPA calculation per semester
    *   Real-time CGPA computation across all semesters
    *   Grade point system with customizable scales
*   **Academic Reports:** Comprehensive reporting system with student performance analytics

### **Advanced Attendance System**
*   **Face Recognition Technology:** 
    *   AI-powered facial recognition for automated attendance marking
    *   High-accuracy face matching with confidence scoring
    *   Support for multiple enrollment photos per user
*   **Flexible Attendance Sessions:** 
    *   Class schedule-based attendance sessions
    *   Support for regular, extra, makeup, and exam sessions
    *   Real-time attendance tracking and analytics
*   **Multiple Attendance Marking Methods:**
    *   Automated face recognition marking
    *   Manual teacher-marked attendance
    *   Device-based attendance through kiosks and tablets
*   **Comprehensive Attendance Analytics:**
    *   Subject-wise attendance statistics
    *   Period-wise attendance reports
    *   Attendance rate calculations and trending

### **Device Management System**
*   **Multi-Device Support:** 
    *   Attendance kiosks, IP cameras, and tablet devices
    *   Device location and department mapping
    *   Network configuration management
*   **Real-time Device Monitoring:**
    *   Device heartbeat tracking and status monitoring
    *   Comprehensive device logging system
    *   Error tracking and maintenance scheduling
*   **Device Dashboard:** Central monitoring with device statistics and health metrics

### **Modern Frontend Interface**
*   **React.js with TypeScript:** Modern, responsive user interface
*   **Role-based Dashboards:** Customized interfaces for each user role
*   **Real-time Data:** Live updates with React Query integration
*   **Component Library:** Shadcn UI for consistent design system
*   **Mobile Responsive:** Optimized for all device sizes

## 🛠 Technology Stack

### **Backend**
*   **Framework:** Django 4.2, Django REST Framework
*   **Database:** PostgreSQL with optimized queries
*   **Authentication:** JWT tokens with djangorestframework-simplejwt
*   **Face Recognition:** face-recognition, dlib, OpenCV
*   **API Documentation:** drf-yasg (Swagger/OpenAPI)
*   **Background Tasks:** Celery with Redis
*   **Production:** Gunicorn, Whitenoise, Docker support

### **Frontend**
*   **Framework:** React 18 with TypeScript
*   **Build Tool:** Vite for fast development and building
*   **UI Library:** Shadcn UI with Tailwind CSS
*   **State Management:** React Query for server state, Context API for client state
*   **Routing:** React Router v6 with protected routes
*   **Forms:** React Hook Form with Zod validation
*   **HTTP Client:** Axios with interceptors for API communication

### **Database Schema**
*   **User Management:** Custom user model with role-based permissions
*   **Academic Structure:** Departments, Academic Years, Subjects, and Enrollments
*   **Attendance System:** Sessions, Records, and Analytics
*   **Device Integration:** Device management with logging and monitoring
*   **Face Recognition:** Secure face encoding storage with encryption

## 🚀 Getting Started

### Prerequisites

*   **Backend:** Python 3.11+, PostgreSQL 13+, Redis 6+
*   **Frontend:** Node.js 18+, npm/yarn/bun
*   **System:** OpenCV dependencies for face recognition
*   **Optional:** Docker for containerized deployment

### Backend Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sudhan2515/College-Management-System.git
    cd College-Management-System/Backend
    ```

2.  **Create and activate virtual environment:**
    ```bash
    python -m venv fr
    # Windows
    fr\Scripts\activate
    # Linux/Mac
    source fr/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create `.env` file in `Backend/attendance_system/`:
    ```env
    SECRET_KEY=your-secret-key-here
    DEBUG=True
    
    # Database Configuration (Supabase/PostgreSQL)
    SUPABASE_DB_HOST=your-host
    SUPABASE_DB_NAME=postgres
    SUPABASE_DB_USER=your-username
    SUPABASE_DB_PASSWORD=your-password
    SUPABASE_DB_PORT=5432
    
    # Or use single DATABASE_URL
    DATABASE_URL=postgresql://user:password@host:port/dbname
    
    # Redis Configuration
    REDIS_URL=redis://localhost:6379/0
    
    # CORS Settings
    CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
    ```

5.  **Run database migrations:**
    ```bash
    cd attendance_system
    python manage.py migrate
    ```

6.  **Create superuser:**
    ```bash
    python manage.py createsuperuser
    ```

7.  **Start development server:**
    ```bash
    python manage.py runserver
    ```

### Frontend Installation

1.  **Navigate to frontend directory:**
    ```bash
    cd Frontend/face-smart
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    # or
    bun install
    ```

3.  **Start development server:**
    ```bash
    npm run dev
    # or
    bun dev
    ```

4.  **Access the application:**
    *   Frontend: `http://localhost:8080`
    *   Backend API: `http://localhost:8000`
    *   API Documentation: `http://localhost:8000/swagger/`

## 📁 Project Structure

```
College-Management-System/
├── Backend/
│   ├── attendance_system/          # Django project root
│   │   ├── users/                  # User management & authentication
│   │   │   ├── models.py          # User, Student, Teacher, Admin models
│   │   │   ├── views.py           # API views and authentication
│   │   │   ├── serializers.py     # DRF serializers
│   │   │   ├── admin.py           # Django admin customization
│   │   │   └── urls.py            # URL routing
│   │   ├── attendance/             # Attendance management system
│   │   │   ├── models.py          # Attendance, Session, Schedule models
│   │   │   ├── views.py           # Attendance API endpoints
│   │   │   └── urls.py            # Attendance URL patterns
│   │   ├── devices/                # Device management system
│   │   │   ├── models.py          # Device, DeviceLog models
│   │   │   ├── views.py           # Device monitoring APIs
│   │   │   └── urls.py            # Device URL patterns
│   │   ├── attendance_system/      # Main Django settings
│   │   │   ├── settings.py        # Project configuration
│   │   │   ├── urls.py            # Main URL configuration
│   │   │   └── wsgi.py            # WSGI application
│   │   ├── media/                  # User uploaded files
│   │   │   ├── student_profiles/   # Student profile pictures
│   │   │   └── teacher_profiles/   # Teacher profile pictures
│   │   ├── logs/                   # Application logs
│   │   └── static/                 # Static files
│   ├── Face/                       # Face recognition utilities
│   │   └── Enrollment.py          # Face enrollment logic
│   └── requirements.txt            # Python dependencies
├── Frontend/
│   └── face-smart/                 # React TypeScript application
│       ├── src/
│       │   ├── components/         # Reusable UI components
│       │   │   └── ui/            # Shadcn UI components
│       │   ├── pages/             # Application pages/screens
│       │   │   ├── Dashboard.tsx  # Role-based dashboards
│       │   │   ├── Login.tsx      # Authentication page
│       │   │   ├── AttendanceView.tsx    # Student attendance view
│       │   │   ├── AttendanceReport.tsx  # Teacher/Admin reports
│       │   │   ├── ManageUsers.tsx       # User management
│       │   │   ├── ClassManagement.tsx   # Class scheduling
│       │   │   ├── FaceEnrollment.tsx    # Face recognition setup
│       │   │   └── PhotoApproval.tsx     # Photo approval workflow
│       │   ├── contexts/          # React contexts
│       │   │   └── AuthContext.tsx # Authentication state
│       │   ├── lib/               # Utility libraries
│       │   │   ├── api.ts         # API client configuration
│       │   │   ├── mockApi.ts     # Mock API for development
│       │   │   └── utils.ts       # Utility functions
│       │   ├── types/             # TypeScript type definitions
│       │   │   └── auth.ts        # Authentication types
│       │   └── hooks/             # Custom React hooks
│       ├── public/                # Static assets
│       ├── package.json           # Node.js dependencies
│       └── vite.config.ts         # Vite configuration
└── README.md                      # Project documentation
```

## 🔧 API Endpoints Overview

### Authentication APIs
*   `POST /api/users/login/` - User authentication
*   `POST /api/users/logout/` - User logout
*   `POST /api/users/password-change/` - Password change

### User Management APIs
*   `GET/POST /api/users/students/` - Student management
*   `GET/POST /api/users/teachers/` - Teacher management
*   `GET/POST /api/users/departments/` - Department management
*   `GET /api/users/academic-years/` - Academic year management

### Academic APIs
*   `GET/POST /api/users/subjects/` - Subject management
*   `GET/POST /api/users/marks/` - Student marks management
*   `GET /api/users/student-report/{student_id}/` - Academic reports
*   `POST /api/users/calculate-sgpa/{student_id}/{semester}/` - SGPA calculation

### Attendance APIs
*   `GET/POST /api/attendance/sessions/` - Attendance sessions
*   `POST /api/attendance/mark-face/` - Face recognition attendance
*   `GET /api/attendance/reports/` - Attendance analytics

### Device Management APIs
*   `GET/POST /api/devices/` - Device management
*   `GET /api/devices/logs/` - Device activity logs
*   `GET /api/devices/dashboard/` - Device monitoring dashboard

## 📊 Database Schema Highlights

### Core Models
*   **User Model:** Extended AbstractUser with role-based permissions
*   **Student Model:** Complete student profile with academic information
*   **Teacher Model:** Faculty management with department assignments
*   **Admin Model:** Administrative user management

### Academic Models
*   **Department:** Centralized department management
*   **AcademicYear:** Academic year tracking with current year flags
*   **Subject:** Subject management with credit system
*   **StudentMarks:** Comprehensive marks management with exam types
*   **SemesterResult:** Automated SGPA calculations
*   **StudentCGPA:** Overall CGPA tracking

### Attendance Models
*   **AttendanceSession:** Class session management with scheduling
*   **Attendance:** Individual attendance records with face recognition data
*   **ClassSchedule:** Timetable management system

### Device Models
*   **Device:** IoT device management with network configuration
*   **DeviceLog:** Comprehensive device activity logging

## 🎯 Current Status & Features

### ✅ Completed Features
*   **Backend (Advanced):**
    *   Complete user management with role-based access
    *   Comprehensive academic management system
    *   Advanced attendance tracking with face recognition
    *   Device management and monitoring system
    *   SGPA/CGPA automated calculations
    *   RESTful APIs with Swagger documentation
    *   JWT authentication with token management

*   **Frontend (In Development):**
    *   User authentication and role-based routing
    *   Dashboard interfaces for all user roles
    *   Basic attendance viewing and reporting
    *   User management interfaces
    *   Face enrollment and photo approval workflows
    *   Class management system

### 🚧 Frontend Development Roadmap

**High Priority:**
*   Academic management interfaces (Subjects, Marks, Reports)
*   SGPA/CGPA calculation and display
*   Advanced attendance analytics and visualizations
*   Device monitoring dashboard
*   Real-time notifications system

**Medium Priority:**
*   Bulk operations for user/data management
*   Advanced reporting with charts and graphs
*   Export functionality (PDF, Excel, CSV)
*   Mobile app development (React Native)

**Future Enhancements:**
*   Integration with LMS platforms
*   Advanced analytics and ML insights
*   Parent/Guardian portal
*   Fee management integration
*   Examination management system

## 🔒 Security Features

*   **JWT-based authentication** with refresh token rotation
*   **Role-based access control** with granular permissions
*   **Face recognition security** with encrypted face encodings
*   **API rate limiting** and request validation
*   **CORS configuration** for secure cross-origin requests
*   **Input validation** and sanitization
*   **Secure file upload** with type verification

## 🚀 Deployment

### Development Setup
*   Backend runs on `http://localhost:8000`
*   Frontend runs on `http://localhost:8080`
*   Database: PostgreSQL (local or Supabase)
*   Redis for caching and background tasks

### Production Deployment
*   **Backend:** Django + Gunicorn + Nginx
*   **Frontend:** Static build served via CDN
*   **Database:** PostgreSQL with connection pooling
*   **Caching:** Redis cluster
*   **Storage:** Cloud storage for media files
*   **Monitoring:** Application and device monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

*   **Developer:** [Sudhan2515](https://github.com/sudhan2515)
*   **Repository:** [College-Management-System](https://github.com/sudhan2515/College-Management-System)

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Note:** This is an active development project. The backend is feature-complete while the frontend is being developed to match the backend's comprehensive functionality. See the roadmap above for current development priorities.
