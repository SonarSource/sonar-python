from django.urls import path


class ClassWithViews:
  def view_method(self):  # FN SONARPY-2322
    ...

  def get_urlpatterns(self):
    return [path("something", self.view_method, name="something")]


def some_view(request):  # Noncompliant
    ...

some_url_patterns = [path("somethind_else", some_view, name="something_else")]
