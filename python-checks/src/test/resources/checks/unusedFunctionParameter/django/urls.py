from django.urls import path

from . import views

urlpatterns = [
  path('view_func1', views.view_func1, name='view_func1'),
  path('view_func2', views.view_func2, name='view_func2'),
  path('view_func3', views.view_func3, name='view_func3'),
]
