import typing as t
from . import typing as ft
from .globals import current_app as current_app, request as request
from typing import Any

http_method_funcs: Any

class View:
    methods: t.ClassVar[t.Optional[t.Collection[str]]]
    provide_automatic_options: t.ClassVar[t.Optional[bool]]
    decorators: t.ClassVar[t.List[t.Callable]]
    init_every_request: t.ClassVar[bool]
    def dispatch_request(self) -> ft.ResponseReturnValue: ...
    @classmethod
    def as_view(cls, name: str, *class_args: t.Any, **class_kwargs: t.Any) -> ft.RouteCallable: ...

class MethodView(View):
    def __init_subclass__(cls, **kwargs: t.Any) -> None: ...
    def dispatch_request(self, **kwargs: t.Any) -> ft.ResponseReturnValue: ...
