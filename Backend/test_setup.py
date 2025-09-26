#!/usr/bin/env python
"""
Test Django setup and check for issues
"""
import os
import sys

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')

try:
    import django
    django.setup()
    print("✅ Django setup successful!")
    
    # Test imports
    from users.models import User, Admin
    print("✅ Models imported successfully!")
    
    # Test database connection
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if result:
            print("✅ Database connection successful!")
    
    # Check migrations
    from django.core.management import execute_from_command_line
    print("\n📋 Checking migrations...")
    execute_from_command_line(['manage.py', 'showmigrations'])
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()