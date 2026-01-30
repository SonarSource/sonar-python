from django.something import receiver
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@receiver(some_signal)
def my_handler(sender, **kwargs):
    ...

@receiver(some_signal)
@csrf_exempt
def my_handler(sender, **kwargs):
    ...

