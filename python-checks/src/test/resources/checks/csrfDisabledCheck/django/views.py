from django.views.decorators.csrf import csrf_exempt

@csrf_exempt # Noncompliant {{Disabling CSRF protection is dangerous.}}
#^^^^^^^^^^^
def csrftest2(request):
    pass
