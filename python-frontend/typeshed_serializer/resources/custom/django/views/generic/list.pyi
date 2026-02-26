from django.views.generic.base import View

class ListView(View):
    model: type
    def get_context_data(self, **kwargs): ...
