import typing as t
from . import json as json
from .globals import current_app as current_app
from typing import Any
from werkzeug.routing import Rule as Rule
from werkzeug.wrappers import Request as RequestBase, Response as ResponseBase

class Request(RequestBase):
    json_module: t.Any
    url_rule: t.Optional[Rule]
    view_args: t.Optional[t.Dict[str, t.Any]]
    routing_exception: t.Optional[Exception]
    @property
    def max_content_length(self) -> t.Optional[int]: ...
    @property
    def endpoint(self) -> t.Optional[str]: ...
    @property
    def blueprint(self) -> t.Optional[str]: ...
    @property
    def blueprints(self) -> t.List[str]: ...
    def on_json_loading_failed(self, e: t.Optional[ValueError]) -> t.Any: ...

class Response(ResponseBase):
    default_mimetype: str
    json_module: Any
    autocorrect_location_header: bool
    @property
    def max_cookie_size(self) -> int: ...
