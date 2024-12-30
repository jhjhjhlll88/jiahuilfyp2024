import uuid
from django.db.models import Q
from django.db import models
from django.utils.translation import gettext_lazy as _
from .utils import getId, generate_ticket_no, convert_hour
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np


# Constants for choices
ISSUE_TYPE_CHOICES = [
    ('Signature', "SIGNATURE"),
    ('Enrollment', "ENROLLMENT"),
    ('Transcript', "TRANSCRIPT"),
    ('Scholarship', "SCHOLARSHIP"),
    ('Examination', "EXAMINATION"),
    ('Other', "OTHER")
]

COMPLAINT_STATUS_CHOICES = [
    ('Unassigned', "UNASSIGNED"),
    ('Assigned', "ASSIGNED"),
    ('Rejected', "REJECTED"),
    ('Completed', "COMPLETED")
]

USER_TYPE_CHOICES = [
    ('Postgraduate', "POSTGRADUATE"),
    ('Undergraduate', "UNDERGRADUATE")
]


ADDRESS_TYPE_CHOICES = [
    ('Mr', "MR"),
    ('Miss', "MISS"),
    ('Mdm', "MDM")
]

PRIO_TYPE_CHOICES = [
    ('Very Urgent', "VERY URGENT"),
    ('Urgent', "URGENT"),
    ('Not Urgent', "NOT URGENT")
]

# Path to the saved BERT model and tokenizer
model_path = '/Users/jiahui/helpdesk/MainApp/bert_issue_classifier'

# Load the trained BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)


class Issues(models.Model):
    """
    Model to represent an issue or complaint raised by users.
    """
    id = models.CharField(editable=False, primary_key=True, max_length=255)  # Unique primary key
    issue_type = models.CharField(
        default=ISSUE_TYPE_CHOICES[0][0], choices=ISSUE_TYPE_CHOICES, max_length=15)
    status = models.CharField(
        default=COMPLAINT_STATUS_CHOICES[0][0], choices=COMPLAINT_STATUS_CHOICES, max_length=15)
    ticket_no = models.CharField(max_length=50, unique=True)
    emp_email = models.EmailField(_('email address'), null=True, blank=True)
    emp_name = models.CharField(max_length=50)
    emp_phone = models.CharField(max_length=15)
    emp_address = models.CharField(
        default=ADDRESS_TYPE_CHOICES[0][0], choices=ADDRESS_TYPE_CHOICES, max_length=255)
    emp_user_type = models.CharField(
        default=USER_TYPE_CHOICES[0][0], choices=USER_TYPE_CHOICES, max_length=20)
    description = models.TextField()
    priority = models.CharField(
        default=PRIO_TYPE_CHOICES[0][0], choices=PRIO_TYPE_CHOICES, max_length=80)
    assign_name = models.CharField(max_length=50, null=True)
    assign_phone = models.CharField(max_length=15, null=True)
    issue_date = models.DateTimeField()
    resolved_date = models.DateTimeField(null=True)
    assigned_date = models.DateTimeField(null=True)
    rejected_date = models.DateTimeField(null=True)
    rejected_reason = models.TextField(null=True)
    completed_reason = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=True)

    # Manager for the model
    objects = models.Manager()

    # Custom properties to calculate time durations
    @property
    def raised_assigned_hour(self):
        return self._calculate_duration(self.created_at, self.assigned_date)

    @property
    def assigned_resolved_hour(self):
        return self._calculate_duration(self.assigned_date, self.resolved_date)

    @property
    def assigned_rejected_hour(self):
        return self._calculate_duration(self.assigned_date, self.rejected_date)

    @property
    def raised_resolved_hour(self):
        return self._calculate_duration(self.created_at, self.resolved_date)

    def _calculate_duration(self, start_time, end_time):
        """
        Helper method to calculate the time duration in hours between two timestamps.
        """
        if start_time and end_time:
            total_seconds = (end_time - start_time).total_seconds()
            return convert_hour(total_seconds)
        return '0'

    def classify_issue_type(self):
        """
        Classify the issue based on the description using the pre-trained BERT model.
        """
        # Tokenize the description
        inputs = tokenizer(self.description, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Predict the issue type using the model
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        # Map the predicted index to the issue type
        issue_type_mapping = {
            0: 'Enrollment',
            1: 'Signature',
            2: 'Scholarship',
            3: 'Transcript',
            4: 'Examination',
            5: 'Other'
        }

        # Return the corresponding issue type label
        return issue_type_mapping.get(predicted_class, 'Other')

    def save(self, *args, **kwargs):
        """
        Custom save method to generate unique identifiers and ticket numbers
        before saving the instance, and to classify the issue type.
        """
        # Classify issue type before saving
        self.issue_type = self.classify_issue_type()

        # Set a unique ID for the issue if not already set
        if not self.id:
            self.id = getId('issue')
            while Issues.objects.filter(id=self.id).exists():
                self.id = getId('issue')

        # Generate unique ticket number if not already set
        if not self.ticket_no:
            self.ticket_no = self._generate_ticket_number()

        super(Issues, self).save(*args, **kwargs)

    def _generate_ticket_number(self):
        """
        Generates a ticket number ensuring uniqueness based on existing count.
        """
        total_issues = Issues.objects.count()
        new_ticket_no = generate_ticket_no(str(total_issues + 1), 'FSKTM')

        # Ensure ticket number uniqueness
        while Issues.objects.filter(ticket_no=new_ticket_no).exists():
            total_issues += 1
            new_ticket_no = generate_ticket_no(str(total_issues + 1), 'FSKTM')

        return new_ticket_no
