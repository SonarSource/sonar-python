from django.urls import path

from . import views
from . import views_decorator_fqn_null

urlpatterns = [
  path('sensitive1', views.sensitive1, name='sensitive1'),
  path('sensitive2', views.sensitive2, name='sensitive2'),
  path('compliant_require_http_methods', views.compliant_require_http_methods, name='compliant_require_http_methods'),
  path('compliant_require_http_methods2', views.compliant_require_http_methods2, name='compliant_require_http_methods2'),
  path('compliant_require_http_methods3', views.compliant_require_http_methods3, name='compliant_require_http_methods3'),
  path('compliant_require_http_methods4', views.compliant_require_http_methods4, name='compliant_require_http_methods4'),
  path('compliant_require_http_methods5', views.compliant_require_http_methods5, name='compliant_require_http_methods5'),
  path('compliant_require_http_methods6', views.compliant_require_http_methods6, name='compliant_require_http_methods6'),
  path('compliant', views_decorator_fqn_null.compliant, name='compliant'),
  path('compliant_require_post', views.compliant_require_post, name='compliant_require_post'),
  path('compliant_require_get', views.compliant_require_get, name='compliant_require_get'),
  path('compliant_require_safe', views.compliant_require_safe, name='compliant_require_safe'),
  path('sensitive_other_dec', views.sensitive_other_dec, name='sensitive_other_dec'),
]
