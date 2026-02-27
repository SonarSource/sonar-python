from typing import Any, Dict, List, Optional, Tuple, Type

from django.core.paginator import Paginator
from django.db.models import Model
from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View

class MultipleObjectMixin(ContextMixin):
    allow_empty: bool
    queryset: Optional[Any]
    model: Optional[Type[Model]]
    paginate_by: Optional[int]
    paginate_orphans: int
    context_object_name: Optional[str]
    paginator_class: Type[Paginator]
    page_kwarg: str
    ordering: Optional[Any]
    object_list: Any
    def get_queryset(self) -> Any: ...
    def get_ordering(self) -> Optional[Any]: ...
    def paginate_queryset(
        self, queryset: Any, page_size: int
    ) -> Tuple[Paginator, Any, Any, bool]: ...
    def get_paginate_by(self, queryset: Any) -> Optional[int]: ...
    def get_paginator(
        self,
        queryset: Any,
        per_page: int,
        orphans: int = ...,
        allow_empty_first_page: bool = ...,
        **kwargs: Any,
    ) -> Paginator: ...
    def get_paginate_orphans(self) -> int: ...
    def get_allow_empty(self) -> bool: ...
    def get_context_object_name(self, object_list: Any) -> Optional[str]: ...
    def get_context_data(
        self, *, object_list: Optional[Any] = ..., **kwargs: Any
    ) -> Dict[str, Any]: ...

class BaseListView(MultipleObjectMixin, View):
    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse: ...

class MultipleObjectTemplateResponseMixin(TemplateResponseMixin):
    template_name_suffix: str
    def get_template_names(self) -> List[str]: ...

class ListView(MultipleObjectTemplateResponseMixin, BaseListView):
    pass
