from django.dispatch import receiver
from django.views.decorators.csrf import csrf_exempt

def custom_decorator(f):
    return f

def another_decorator(f):
    return f

@csrf_exempt
@receiver(some_signal) # Noncompliant {{Move this '@receiver' decorator to the top of the other decorators.}}
def my_handler(sender, **kwargs):
    ...

@receiver(some_signal)
@csrf_exempt
def compliant_handler(sender, **kwargs):
    ...

@custom_decorator
@csrf_exempt
@receiver(some_signal) # Noncompliant
def multiple_before_receiver(sender, **kwargs):
    ...

@csrf_exempt
@receiver(some_signal) # Noncompliant
@custom_decorator
def receiver_in_middle(sender, **kwargs):
    ...

@receiver(some_signal)
@csrf_exempt
@custom_decorator
def receiver_first_multiple(sender, **kwargs):
    ...

@receiver(some_signal)
def single_receiver(sender, **kwargs):
    ...

@csrf_exempt
@custom_decorator
def no_receiver(sender, **kwargs):
    ...
