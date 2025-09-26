from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('users', '0002_user_must_change_password'),
    ]

    operations = [
        migrations.CreateModel(
            name='Subject',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('code', models.CharField(max_length=20, unique=True)),
                ('semester', models.IntegerField()),
                ('credits', models.DecimalField(decimal_places=1, max_digits=3)),
                ('has_practical', models.BooleanField(default=False)),
                ('is_active', models.BooleanField(default=True)),
                ('department', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='subjects', to='users.department')),
            ],
            options={
                'unique_together': {('department', 'semester', 'code')},
            },
        ),
        migrations.CreateModel(
            name='ExamType',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(choices=[('internal1', 'Internal Exam 1'), ('internal2', 'Internal Exam 2'), ('practical', 'Practical Exam'), ('semester', 'Semester End Exam')], max_length=20, unique=True)),
                ('display_name', models.CharField(max_length=50)),
                ('max_marks', models.IntegerField(default=100)),
                ('weightage_percentage', models.DecimalField(decimal_places=2, default=100.0, max_digits=5)),
            ],
        ),
        migrations.CreateModel(
            name='StudentMarks',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('marks_obtained', models.DecimalField(decimal_places=2, max_digits=5)),
                ('max_marks', models.IntegerField(default=100)),
                ('entered_date', models.DateTimeField(auto_now_add=True)),
                ('updated_date', models.DateTimeField(auto_now=True)),
                ('academic_year', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.academicyear')),
                ('entered_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='users.teacher')),
                ('exam_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.examtype')),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='marks', to='users.student')),
                ('subject', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.subject')),
            ],
            options={
                'unique_together': {('student', 'subject', 'exam_type', 'academic_year')},
            },
        ),
        migrations.CreateModel(
            name='SemesterResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('semester', models.IntegerField()),
                ('sgpa', models.DecimalField(blank=True, decimal_places=2, max_digits=4, null=True)),
                ('total_credits', models.DecimalField(decimal_places=1, default=0, max_digits=5)),
                ('calculated_date', models.DateTimeField(auto_now=True)),
                ('academic_year', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.academicyear')),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='semester_results', to='users.student')),
            ],
            options={
                'unique_together': {('student', 'semester', 'academic_year')},
            },
        ),
        migrations.CreateModel(
            name='StudentCGPA',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cgpa', models.DecimalField(blank=True, decimal_places=2, max_digits=4, null=True)),
                ('total_credits', models.DecimalField(decimal_places=1, default=0, max_digits=6)),
                ('semesters_completed', models.IntegerField(default=0)),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('student', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='cgpa_record', to='users.student')),
            ],
        ),
    ]