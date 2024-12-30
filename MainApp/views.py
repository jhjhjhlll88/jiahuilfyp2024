import torch
from transformers import BertForSequenceClassification, BertTokenizer  
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth import login as log_in
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.conf import settings
from django.db.models import Q
from django.db import transaction
from MainApp.models import Issues
from datetime import datetime
import json
import re
from django.views.decorators.csrf import csrf_exempt
from .utils import Syserror




UserModel = get_user_model()

# Choices for validation
ISSUE_TYPE_CHOICES = ['Signature', 'Enrollment', 'Transcript', 'Scholarship', 'Examination', 'Other']
ADDRESS_CHOICES = ['Mr', 'Miss', 'Mdm']
USER_CHOICES = ['Postgraduate', 'Undergraduate']
PRIORITY_CHOICES = ['Very Urgent', 'Urgent', 'Not Urgent']

# Path to the saved BERT model and tokenizer
model_path = '/Users/jiahui/helpdesk/MainApp/bert_issue_classifier'

# Load the trained BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)


def HomeView(request):
    """
    Home view that renders the homepage template.
    """
    return render(request, 'MainApp/index.html')


def RaiseTicketView(request):
    """
    View for rendering the ticket raising form.
    """
    return render(request, 'MainApp/ticket_raise.html')

# View to render the chatbot page (front-end component)
def chatbot(request):
    """Render the chatbot page where the user can interact with the bot."""
    return render(request, 'MainApp/chatbot.html')

# View to process the query from the user and return the chatbot's response
@csrf_exempt
def query_chatbot(request):
    """
    Handle the chatbot query.
    Expects a POST request with a 'query' parameter containing the user's question.
    """
    if request.method == "POST":
        query = request.POST.get("query", "")
        
        if not query:
            return JsonResponse({"success": False, "message": "Query cannot be empty."})
        
        try:
            # Query the Chroma vector store for relevant responses
            results = query_chroma_vectorstore(query)
            
            # Return the results as a JSON response
            return JsonResponse({"success": True, "results": [result["text"] for result in results]})
        
        except Exception as e:
            return JsonResponse({"success": False, "message": f"Error processing query: {e}"})
    
    return JsonResponse({"success": False, "message": "Invalid request method."})



@csrf_exempt
def raised_issue(request):
    """
    Endpoint to handle raised issue requests via POST.
    """
    if request.method != 'POST':
        return JsonResponse({"success": False, "message": "POST method required."})

    try:
        # Parse incoming data
        data = json.loads(request.body)
        validation_errors = validate_ticket_data(data)
        
        if validation_errors:
            return JsonResponse({"success": False, "message": validation_errors})

        # Classify the issue type based on the description using the AI model
        issue_type = classify_issue_type(data["description"])

        # If all validations pass, create the issue
        with transaction.atomic():
            issue = Issues.objects.create(
                emp_name=data["name"],
                emp_email=data.get("email", None),
                emp_phone=data["phone"],
                emp_address=data["address"],
                emp_user_type=data["user_type"],
                issue_type=issue_type,  # AI-generated issue type
                priority=data["priority"],
                description=data["description"],
                issue_date=datetime.now(),
            )

            # Send email to the user
            if data.get("email"):
                subject = "Your Ticket Submission Confirmation"
                message = (
                    f"Dear {data['name']},\n\n"
                    f"Thank you for submitting your issue. Here are the details of your inquiry:\n\n"
                    f"Ticket Number: {issue.ticket_no}\n"
                    f"Issue Type: {issue.issue_type}\n"
                    f"Priority: {issue.priority}\n"
                    f"Description: {issue.description}\n\n"
                    f"We will address your issue as soon as possible.\n\n"
                    f"Thank you,\nSupport Team"
                )

                # Sending email using settings configured in settings.py
                send_mail(
                    subject,
                    message,
                    settings.EMAIL_HOST_USER,  # This ensures the correct sender address
                    [data["email"]],
                    fail_silently=False,  # You can set to True if you want to suppress errors
                )

            return JsonResponse({
                "success": True,
                "message": "Issue raised successfully.",
                "ticket_number": issue.ticket_no
            })

    except Exception as e:
        Syserror(e)
        return JsonResponse({"success": False, "message": f"Server error: {e}"})


def validate_ticket_data(data):
    """
    Helper function to validate the incoming ticket data.
    Returns validation errors or None if valid.
    """
    # Ensure all required fields are present
    required_fields = ["name", "phone", "address", "user_type", "priority", "description", "email"]
    if not all([data.get(field) for field in required_fields]):
        return "All fields are required, including email."

    # Validate phone number format
    if not data["phone"].isdigit() or len(data["phone"]) != 10:
        return "Phone number must be 10 digits."

    # Validate email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zAZ0-9.-]+\.[a-zA-Z]{2,}$', data["email"]):
        return "Invalid email format."

    # Validate address type
    if data["address"] not in ADDRESS_CHOICES:
        return "Invalid address type."

    # Validate user type
    if data["user_type"] not in USER_CHOICES:
        return "Invalid user type."
    
    # Validate priority type
    if data["priority"] not in PRIORITY_CHOICES:
        return "Invalid priority type."

    return None


def classify_issue_type(description):
    """
    Classify the issue based on the description using the pre-trained PyTorch model.
    """
    # Tokenize the description
    inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=50)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Predict the issue type using the model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Map the predicted index to issue type
    issue_type_mapping = {
        0: 'Enrollment',
        1: 'Signature',
        2: 'Scholarship',
        3: 'Transcript',
        4: 'Examination',
        5: 'Other'
    }

    return issue_type_mapping.get(predicted_class, 'Other')  # Default to 'Other' if not found


def TicketStatusView(request):
    """
    View to check the status of a ticket based on the ticket number, employee email,
    name, or phone number.
    """
    search_term = request.GET.get("search_term")
    issue = None

    if search_term:
        # Try to search by ticket number
        issue = Issues.objects.filter(ticket_no=search_term).first()
        
        # If no result, search by emp_email
        if not issue:
            issue = Issues.objects.filter(emp_email=search_term).first()

        # If still no result, search by emp_name
        if not issue:
            issue = Issues.objects.filter(emp_name__icontains=search_term).first()

        # If still no result, search by emp_phone
        if not issue:
            issue = Issues.objects.filter(emp_phone=search_term).first()

        # If no issues found, display an error message
        if not issue:
            messages.error(request, "No issue found with the given details. Please try again with a valid ticket number, email, name, or phone number.")
    
    return render(request, 'MainApp/ticket_status.html', {'issue': issue, 'search_term': search_term})


def send_issue_email(issue, subject, message):
    try:
        send_mail(
            subject,
            message,
            settings.EMAIL_HOST_USER,
            [issue.emp_email],
            fail_silently=False,
        )
    except Exception as e:
        messages.warning(issue.request, f"Email notification failed: {e}")


def update_issue_status(request, issue_id):
    if request.method == "POST":
        try:
            issue = Issues.objects.get(id=issue_id)

            new_status = request.POST.get("status")
            completed_reason = request.POST.get("completed_reason")
            rejected_reason = request.POST.get("rejected_reason")

            if issue.status in ["Completed", "Rejected"]:
                messages.error(request, "This issue has already been resolved or rejected.")
                return redirect("manage_issues")

            if new_status == "Assigned":
                issue.status = "Assigned"
                issue.assigned_date = datetime.now()

                if issue.emp_email:
                    send_issue_email(
                        issue,
                        f"Inquiry Assigned (Ticket #{issue.ticket_no})",
                        (
                            f"Dear {issue.emp_name},\n\n"
                            f"Your inquiry has been assigned.\n"
                            f"Ticket Number: {issue.ticket_no}\n"
                            f"Current Status: Assigned\n\n"
                            "Best regards,\nSupport Team"
                        ),
                    )
            elif new_status == "Completed":
                if not completed_reason:
                    messages.error(request, "Completion reason is required.")
                    return redirect("manage_issues")
                
                issue.status = "Completed"
                issue.resolved_date = datetime.now()
                issue.completed_reason = completed_reason

                if issue.emp_email:
                    send_issue_email(
                        issue,
                        f"Inquiry Completed (Ticket #{issue.ticket_no})",
                        (
                            f"Dear {issue.emp_name},\n\n"
                            f"Your inquiry has been resolved.\n"
                            f"Completion Reason: {completed_reason}\n\n"
                            "Best regards,\nSupport Team"
                        ),
                    )
            elif new_status == "Rejected":
                if not rejected_reason:
                    messages.error(request, "Rejection reason is required.")
                    return redirect("manage_issues")
                
                issue.status = "Rejected"
                issue.rejected_date = datetime.now()
                issue.rejected_reason = rejected_reason

                if issue.emp_email:
                    send_issue_email(
                        issue,
                        f"Inquiry Rejected (Ticket #{issue.ticket_no})",
                        (
                            f"Dear {issue.emp_name},\n\n"
                            f"Your inquiry has been rejected.\n"
                            f"Rejection Reason: {rejected_reason}\n\n"
                            "Best regards,\nSupport Team"
                        ),
                    )
            else:
                messages.error(request, "Invalid status.")
                return redirect("manage_issues")

            issue.save()
            messages.success(request, "Issue status updated and notification sent to the user.")
            return redirect("manage_issues")

        except Issues.DoesNotExist:
            messages.error(request, "The issue does not exist.")
            return redirect("manage_issues")

        except Exception as e:
            messages.error(request, f"Error updating issue status: {e}")
            return redirect("manage_issues")

    messages.error(request, "Invalid request method.")
    return redirect("manage_issues")


def loginView(request):
    """
    View for logging in users.
    """
    if request.method == "POST":
        try:
            email = request.POST.get("email")
            password = request.POST.get("password")

            if not email or not password:
                messages.error(request, "Email and Password are required.")
                return redirect(reverse('login'))

            user = UserModel.objects.filter(Q(email=email) | Q(username=email)).first()
            if user and user.check_password(password):
                log_in(request, user)
                return redirect(reverse('dashboard'))

            messages.error(request, "Invalid credentials.")
            return redirect(reverse('login'))

        except Exception as e:
            messages.error(request, f"Server error: {e}")
            return redirect(reverse('login'))
    else:
        return render(request, 'MainApp/login.html')


def assign_issue(request, issue_id):
    try:
        # Fetch the issue by ID
        issue = get_object_or_404(Issues, id=issue_id)

        if request.method == 'POST':
            # Get the form data
            assign_name = request.POST.get('assign_name')
            assign_phone = request.POST.get('assign_phone')

            # Assign the issue
            issue.assign_name = assign_name
            issue.assign_phone = assign_phone
            issue.status = 'Assigned'
            issue.assigned_date = datetime.now()
            issue.save()

            messages.success(request, f"Issue #{issue.ticket_no} has been assigned successfully.")

            return redirect('manage_issues')  # Adjust this to the correct view after assignment

    except Issues.DoesNotExist:
        messages.error(request, "Issue not found.")
        return redirect('manage_issues')  # Redirect to the manage issues page if the issue doesn't exist

    return render(request, 'assign_issue.html', {'issue': issue})
