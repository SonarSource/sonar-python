from django.views.decorators.http import require_safe

if 42:
    def require_safe(): ...

@require_safe
def compliant(request): # Compliant
  ...
