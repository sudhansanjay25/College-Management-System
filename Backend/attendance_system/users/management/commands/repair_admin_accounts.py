from django.core.management.base import BaseCommand
from django.db import transaction, IntegrityError
from django.contrib.auth import get_user_model
from users.models import Admin

try:
    from django.contrib.admin.models import LogEntry
except Exception:  # pragma: no cover
    LogEntry = None

User = get_user_model()

class Command(BaseCommand):
    help = (
        "Inspect and repair orphan admin User accounts (username starts with ADM, user_type=admin, no Admin row).\n"
        "Options: --convert (create Admin rows), --deactivate (rename + disable), --delete (try full removal)."
    )

    def add_arguments(self, parser):
        parser.add_argument('--convert', action='store_true', help='Create Admin objects for orphans')
        parser.add_argument('--deactivate', action='store_true', help='Deactivate + rename orphan users')
        parser.add_argument('--delete', action='store_true', help='Attempt hard delete orphan users (falls back to deactivate on FK restriction)')
        parser.add_argument('--dry-run', action='store_true', help='Show what would happen without changing data')

    def handle(self, *args, **opts):
        convert = opts['convert']
        deactivate = opts['deactivate']
        delete = opts['delete']
        dry = opts['dry_run']

        orphans = User.objects.filter(username__startswith='ADM', user_type='admin', admin__isnull=True)
        count = orphans.count()
        self.stdout.write(self.style.NOTICE(f"Found {count} orphan admin user(s)."))
        if not count:
            return

        for u in orphans.order_by('username'):
            self.stdout.write(f"- {u.username} (active={u.is_active})")

        if dry:
            self.stdout.write(self.style.WARNING("Dry run: no changes applied."))
            return

        for u in orphans:
            if convert:
                if not Admin.objects.filter(admin_id=u.username).exists():
                    Admin.objects.create(
                        user=u,
                        admin_id=u.username,
                        first_name=u.first_name or 'Recovered',
                        last_name=u.last_name or '',
                        email=u.email or f"{u.username.lower()}@placeholder.local",
                        role='Recovered'
                    )
                    self.stdout.write(self.style.SUCCESS(f"Converted {u.username} -> Admin row created."))
                else:
                    self.stdout.write(self.style.WARNING(f"Admin row already exists for id {u.username}, skipping convert."))
                continue

            if delete:
                try:
                    with transaction.atomic():
                        # Try to null admin log references if possible
                        if LogEntry is not None and LogEntry._meta.get_field('user').null:
                            LogEntry.objects.filter(user=u).update(user=None)
                        u.delete()
                        self.stdout.write(self.style.SUCCESS(f"Deleted orphan user {u.username}"))
                        continue
                except IntegrityError as e:
                    self.stdout.write(self.style.WARNING(f"Delete blocked for {u.username} ({e}); falling back to deactivate."))
                    # Fall through to deactivate

            if deactivate or delete:  # deactivate path (explicit or fallback)
                old_username = u.username
                if not old_username.startswith('ARCHIVED_'):
                    # Free the ADM prefix space for future IDs
                    u.username = f"ARCHIVED_{old_username}"
                u.is_active = False
                u.save(update_fields=['username', 'is_active'])
                self.stdout.write(self.style.SUCCESS(f"Deactivated orphan user {old_username} (renamed -> {u.username})"))

        self.stdout.write(self.style.SUCCESS("Repair process complete."))
