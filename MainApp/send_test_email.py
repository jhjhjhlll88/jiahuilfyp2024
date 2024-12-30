from django.core.management.base import BaseCommand
from MainApp.views import send_email_directly  # Import your send_email_directly function

class Command(BaseCommand):
    help = "Sends a test email to verify email functionality."

    def handle(self, *args, **kwargs):
        # Replace 'recipient@example.com' with the email you want to test
        recipient_email = "sulamproject2@gmail.com"

        try:
            self.stdout.write("Sending test email...")
            send_email_directly(
                "Test Subject",
                "This is a test email. If you receive it, email functionality works!",
                recipient_email
            )
            self.stdout.write(self.style.SUCCESS(f"Test email successfully sent to {recipient_email}."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to send test email: {e}"))
