from django.utils.deprecation import MiddlewareMixin


class MyMiddleware(MiddlewareMixin):
    def process_request(self, request):  # Compliant - Django middleware hook
        print("logging")

    def process_exception(
        self, request, exception
    ):  # Compliant - Django middleware hook
        print("logging")

    def process_view(
        self, request, view_func, view_args, view_kwargs
    ):  # Compliant - Django middleware hook
        print("logging")

    def process_template_response(
        self, request, response
    ):  # Compliant - Django middleware hook
        print("logging")

    def some_other_method(self, foo):  # Noncompliant
        print("logging")


class NotAMiddleware:
    def process_request(self, request):  # Noncompliant
        print("logging")

    def process_exception(self, request, exception):  # Noncompliant 2
        print("logging")
