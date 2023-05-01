from django.dispatch import receiver as renamed_receiver
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@renamed_receiver(some_signal) # Noncompliant {{Move this '@receiver' decorator to the top of the other decorators.}}
def my_handler(sender, **kwargs):
    ...

@renamed_receiver(some_signal)
@csrf_exempt
def my_handler(sender, **kwargs):
    ...
