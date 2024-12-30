from django.urls import path
from AdminApp import views
urlpatterns = [
    path('logout', views.logoutView, name="admin_logout"),
    path('dashboard', views.dashboardView, name="dashboard"),
    path('manage_tickets', views.manageIssuesView, name="manage_issues"),
    path('tickets_report', views.ticketReportView, name="tickets_report"),
    path('update_issue_status', views.updateIssueStatusView, name="update_issue_status"),
    path('issue_analytics', views.issueAnalyticsView, name="issue_analytics"),
    path('issue_average_time_analytics', views.issueAverageTimeAnalyticsView, name="issue_average_time_analytics"),
    path('ticket_report_average_time', views.ticketReportAverateTimeView, name="ticket_report_average_time"),
    path('change_name', views.changeNameView, name="change_name"),
    path('download_ticket_report/<str:id>', views.download_ticket_report, name="download_ticket_report"),

]
