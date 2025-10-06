from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_add_academic_models'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='student',
            name='face_enrollment_date',
        ),
        migrations.RemoveField(
            model_name='teacher',
            name='face_enrollment_date',
        ),
    ]
