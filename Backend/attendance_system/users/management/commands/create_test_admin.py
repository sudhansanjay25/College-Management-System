from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from users.models import Admin

User = get_user_model()

class Command(BaseCommand):
    help = 'Create a test admin user for login testing'

    def add_arguments(self, parser):
        parser.add_argument('--username', default='testadmin', help='Admin username')
        parser.add_argument('--password', default='testpass123', help='Admin password')
        parser.add_argument('--email', default='admin@test.com', help='Admin email')

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        email = options['email']
        
        # Check if user already exists
        if User.objects.filter(username=username).exists():
            self.stdout.write(
                self.style.WARNING(f'User "{username}" already exists!')
            )
            return
        
        # Create superuser
        try:
            user = User.objects.create_superuser(
                username=username,
                email=email,
                password=password,
                user_type='admin',
                first_name='Test',
                last_name='Admin'
            )
            
            # Create corresponding Admin record
            admin = Admin.objects.create(
                user=user,
                admin_id=f'TEST_ADM_{user.id:03d}',
                first_name='Test',
                last_name='Admin',
                email=email,
                role='Super Admin'
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully created admin user:\n'
                    f'  Username: {username}\n'
                    f'  Password: {password}\n'
                    f'  Admin ID: {admin.admin_id}\n'
                    f'  Email: {email}\n'
                    f'  Can login to Django admin: Yes'
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error creating admin user: {str(e)}')
            )