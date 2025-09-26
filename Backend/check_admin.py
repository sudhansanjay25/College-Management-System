#!/usr/bin/env python
"""
Script to check and fix admin login issues
"""
import os
import django
from django.core.management.base import BaseCommand

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from users.models import User, Admin
from django.contrib.auth import authenticate

def main():
    print("=== Admin Account Checker ===\n")
    
    # Check all users
    print("All Users in database:")
    users = User.objects.all()
    if users:
        for user in users:
            print(f"  Username: {user.username}")
            print(f"  Email: {user.email}")
            print(f"  User Type: {user.user_type}")
            print(f"  Is Staff: {user.is_staff}")
            print(f"  Is Superuser: {user.is_superuser}")
            print(f"  Is Active: {user.is_active}")
            print(f"  Must Change Password: {user.must_change_password}")
            print("  ---")
    else:
        print("  No users found!")
    
    print("\nAll Admin records:")
    admins = Admin.objects.all()
    if admins:
        for admin in admins:
            print(f"  Admin ID: {admin.admin_id}")
            print(f"  Name: {admin.full_name}")
            print(f"  Email: {admin.email}")
            print(f"  Has User: {bool(admin.user)}")
            if admin.user:
                print(f"  User Username: {admin.user.username}")
            print("  ---")
    else:
        print("  No admin records found!")
    
    # Test authentication
    print("\nTesting authentication for common credentials:")
    test_credentials = [
        ('admin24', 'admin123'),
        ('admin', 'admin123'),
        ('superuser', 'admin123'),
    ]
    
    for username, password in test_credentials:
        user = authenticate(username=username, password=password)
        if user:
            print(f"  ✓ {username}/{password} - LOGIN SUCCESS")
            print(f"    Can access admin: {user.is_staff and user.is_superuser}")
        else:
            print(f"  ✗ {username}/{password} - LOGIN FAILED")
    
    # Create a test superuser if none exists
    if not User.objects.filter(is_superuser=True).exists():
        print("\n⚠️  No superuser exists! Creating test superuser...")
        try:
            User.objects.create_superuser(
                username='testadmin',
                email='admin@test.com',
                password='testpass123',
                user_type='admin'
            )
            print("✓ Test superuser created: testadmin/testpass123")
        except Exception as e:
            print(f"✗ Failed to create superuser: {e}")

if __name__ == '__main__':
    main()