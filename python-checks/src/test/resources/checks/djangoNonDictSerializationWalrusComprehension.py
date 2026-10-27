from django.http import JsonResponse

values = [(data := []) for x in range(1)]
JsonResponse(
    data  # Noncompliant {{Use a dictionary object here, or set the "safe" flag to False.}}
)
