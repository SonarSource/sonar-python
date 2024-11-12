from django.views.decorators.http import require_http_methods, require_POST, require_GET, require_safe, other_decorator

def sensitive1(request):  # Noncompliant
  ...

@require_http_methods(["GET", "POST"])  # Noncompliant
def sensitive2(request):
  ...

@require_http_methods(["POST"])  # Compliant
def compliant_require_http_methods(request):
  ...

@require_http_methods(["GET"])  # Compliant
def compliant_require_http_methods2(request):
  ...
@require_http_methods()  # Compliant
def compliant_require_http_methods3(request):
  ...

methods = []
@require_http_methods(*methods)  # Compliant
def compliant_require_http_methods4(request):
    ...

@unknown(["GET", "POST"])
def compliant_require_http_methods5(request): # Noncompliant
    ...

def foo(): ...
@foo(["GET", "POST"])
def compliant_require_http_methods6(request): # Noncompliant
    ...

@require_POST  # Compliant
def compliant_require_post(request):
  ...

@require_GET  # Compliant
def compliant_require_get(request):
  ...

@require_safe  # Compliant
def compliant_require_safe(request):
  ...
@other_decorator
def sensitive_other_dec(request): # Noncompliant
  ...

def other(request):
  ...


from django.urls import path

class SameClassView:
  def chosen_view(self):  # FN SONARPY-2322
    ...

  def get_urlpatterns(self):
    return [path("something", self.chosen_view, name="something")]
