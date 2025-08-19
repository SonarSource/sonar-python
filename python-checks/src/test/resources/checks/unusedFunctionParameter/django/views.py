from django.views.decorators.http import require_http_methods, require_POST, require_GET, require_safe, other_decorator

def not_a_view(request): # Noncompliant
    print("smth")

def view_func1(request):  # Compliant
  print("smth")

@require_http_methods(["GET", "POST"])  # Compliant
def view_func2(request):
    print("smth")

def view_func3(request, unused_parameter):  # Compliant FN, unused_parameter is not used in the function body and is not mandatory
    print("smth")

