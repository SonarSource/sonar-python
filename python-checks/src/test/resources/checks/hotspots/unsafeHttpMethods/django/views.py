from django.views.decorators.http import require_http_methods, require_POST, require_GET, require_safe, other_decorator
from django.contrib.auth.decorators import login_required, permission_required
from some_framework import view_decorator

@require_http_methods(["GET", "POST"])  # Compliant - methods explicitly specified
def compliant_explicit_methods(request):
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

@view_decorator  # Compliant: non-django unresolved import may restrict HTTP methods
def compliant_non_django_import(request):
    ...

@unknown(["GET", "POST"])
def compliant_require_http_methods5(request): # Compliant: Unknown decorator might restrict allowed HTTP verbs
    ...

def foo(): ...
@foo(["GET", "POST"])
def compliant_require_http_methods6(request): # Compliant
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

def compliant_method_eq_check(request):  # Compliant: method check in body
  if request.method == "POST":
    ...

def compliant_method_eq_check_swapped(request):  # Compliant: operands swapped but still a method check
  if "POST" == request.method:
    ...

def compliant_method_in_check(request):  # Compliant: method check in body
  if request.method in ["GET", "POST"]:
    ...

def compliant_method_check_and_condition(request):  # Compliant: method check nested in boolean if condition
  if request.user.is_authenticated and request.method == "POST":
    ...

def compliant_method_check_parenthesized(request):  # Compliant: comparison wrapped in parentheses
  if (request.method == "POST"):
    ...

def compliant_method_check_elif(request):  # Compliant: method check in elif
  if request.user.is_staff:
    return "staff"
  elif request.method == "GET":
    ...

def compliant_method_check_in_nested_scope(request):  # Compliant: outer request captured by nested scope, method is checked
  def inner():
    if request.method == "POST":
      ...
  inner()

def noncompliant_nested_function(request, something_else):  # Noncompliant
  def helper(request):
    if request.method == "POST":
      ...
  helper(something_else)

def noncompliant_lambda(request):  # Noncompliant
  _ = lambda req: req.method == "POST"
  ...

def noncompliant1(request):  # Noncompliant
  ...

@other_decorator
def noncompliant_other_dec(request): # Noncompliant
  ...

@login_required
def noncompliant_login_required(request):  # Noncompliant
  ...

@permission_required("app.permission")
def noncompliant_permission_required(request):  # Noncompliant
  ...

def noncompliant_method_on_other_var(request):  # Noncompliant
  _ = request.session
  ...

def noncompliant_method_on_different_var(request):  # Noncompliant
  if another.method == "POST":
    ...

def noncompliant_other_if_eq_check(request):  # Noncompliant
  if request.user.is_authenticated == True:
    ...

def noncompliant_method_truthy_check(request):  # Noncompliant
  if request.method:
    ...

def noncompliant_method_and_check(request):  # Noncompliant
  if request.method and request.user.is_authenticated:
    ...

def noncompliant_method_only_printed(request):  # Noncompliant
  print(request.method)
  ...

def noncompliant_method_comparison_not_in_if(request):  # Noncompliant
  _ = request.method == "POST"
  ...

def noncompliant_request_used_as_argument(request):  # Noncompliant
  do_something(request)
  ...

# Artificial test for Python 2-style tuple parameters: nonTuple() is empty so request.method cannot be checked
def noncompliant_tuple_param_only((a, b)):  # Noncompliant
  ...

def other(request):
  ...
