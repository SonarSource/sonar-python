from django.dispatch import receiver
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@receiver(some_signal) # Noncompliant {{Move this '@receiver' decorator to the top of the other decorators.}}
def my_handler(sender, **kwargs):
    ...

@receiver(some_signal)
@csrf_exempt
def my_handler(sender, **kwargs):
    ...
