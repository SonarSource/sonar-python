from django.http import JsonResponse

condition = bool()
JsonResponse({} if condition else [])
