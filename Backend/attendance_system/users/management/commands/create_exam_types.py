from django.core.management.base import BaseCommand
from users.models import ExamType


class Command(BaseCommand):
    help = 'Create default exam types for the academic system'

    def handle(self, *args, **options):
        exam_types_data = [
            {
                'name': 'internal1',
                'display_name': 'Internal Exam 1',
                'max_marks': 100,
                'weightage_percentage': 15.00
            },
            {
                'name': 'internal2',
                'display_name': 'Internal Exam 2',
                'max_marks': 100,
                'weightage_percentage': 15.00
            },
            {
                'name': 'practical',
                'display_name': 'Practical Exam',
                'max_marks': 100,
                'weightage_percentage': 20.00
            },
            {
                'name': 'semester',
                'display_name': 'Semester End Exam',
                'max_marks': 100,
                'weightage_percentage': 50.00
            }
        ]

        created_count = 0
        for exam_data in exam_types_data:
            exam_type, created = ExamType.objects.get_or_create(
                name=exam_data['name'],
                defaults={
                    'display_name': exam_data['display_name'],
                    'max_marks': exam_data['max_marks'],
                    'weightage_percentage': exam_data['weightage_percentage']
                }
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created exam type: {exam_type.display_name}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Exam type already exists: {exam_type.display_name}')
                )

        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {created_count} new exam types')
        )