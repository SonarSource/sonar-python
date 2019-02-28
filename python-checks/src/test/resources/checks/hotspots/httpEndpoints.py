from django.urls import path, re_path

def declare_views(views):
    return [
        path('', views['index']),  # Noncompliant {{Make sure that exposing this HTTP endpoint is safe here.}}
        re_path(r'^about/[0-9]*/$', views['about']),  # Noncompliant
    ]
