from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin
from django.shortcuts import render


class ProtectView(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        modulename = view_func.__module__
        user = request.user
        if user.is_authenticated:
            if modulename not in ['AdminApp.views', 'django.views.static']:
                return HttpResponseRedirect(reverse("dashboard"))
        else:
            if modulename in ['MainApp.views', 'django.views.static']:
                pass
            else:
                return HttpResponseRedirect(reverse("login"))
