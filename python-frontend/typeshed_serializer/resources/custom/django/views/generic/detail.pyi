from typing import Any, Dict, List, Optional, Type

from django.db.models import Model
from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View

class SingleObjectMixin(ContextMixin):
    model: Optional[Type[Model]]
    queryset: Optional[Any]
    slug_field: str
    context_object_name: Optional[str]
    slug_url_kwarg: str
    pk_url_kwarg: str
    query_pk_and_slug: bool
    object: Any
    def get_object(self, queryset: Optional[Any] = ...) -> Any: ...
    def get_queryset(self) -> Any: ...
    def get_slug_field(self) -> str: ...
    def get_context_object_name(self, obj: Any) -> Optional[str]: ...
    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]: ...

class BaseDetailView(SingleObjectMixin, View):
    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse: ...

class SingleObjectTemplateResponseMixin(TemplateResponseMixin):
    template_name_field: Optional[str]
    template_name_suffix: str
    def get_template_names(self) -> List[str]: ...

class DetailView(SingleObjectTemplateResponseMixin, BaseDetailView):
    pass
