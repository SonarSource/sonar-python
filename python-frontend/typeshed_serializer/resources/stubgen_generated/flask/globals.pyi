import typing as t
from .app import Flask as Flask
from .ctx import AppContext as AppContext, RequestContext as RequestContext, _AppCtxGlobals
from .sessions import SessionMixin as SessionMixin
from .wrappers import Request as Request
from contextvars import ContextVar
from typing import Any

class _FakeStack:
    name: Any
    cv: Any
    def __init__(self, name: str, cv: ContextVar[t.Any]) -> None: ...
    def push(self, obj: t.Any) -> None: ...
    def pop(self) -> t.Any: ...
    @property
    def top(self) -> t.Optional[t.Any]: ...

app_ctx: AppContext
current_app: Flask
g: _AppCtxGlobals
request_ctx: RequestContext
request: Request
session: SessionMixin

def __getattr__(name: str) -> t.Any: ...
