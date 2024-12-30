from django.urls import path
from MainApp import views
urlpatterns = [
    path('', views.HomeView, name="home"),
    path('inquiries_submit', views.RaiseTicketView, name="raise_ticket"),
    path('inquiries_status', views.TicketStatusView, name="ticket_status"),
    path('adminlogin', views.loginView, name="login"),
    path('raised_inquiries', views.raised_issue, name="raised_issue"),
    path('update-inquiries-status/<str:issue_id>/', views.update_issue_status, name='update_issue_status'),
]
