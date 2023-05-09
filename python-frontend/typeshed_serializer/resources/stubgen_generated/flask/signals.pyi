import typing as t
from typing import Any

signals_available: bool

class Namespace:
    def signal(self, name: str, doc: t.Optional[str] = ...) -> _FakeSignal: ...

class _FakeSignal:
    name: Any
    __doc__: Any
    def __init__(self, name: str, doc: t.Optional[str] = ...) -> None: ...
    def send(self, *args: t.Any, **kwargs: t.Any) -> t.Any: ...
    connect: Any
    connect_via: Any
    connected_to: Any
    temporarily_connected_to: Any
    disconnect: Any
    has_receivers_for: Any
    receivers_for: Any

template_rendered: Any
before_render_template: Any
request_started: Any
request_finished: Any
request_tearing_down: Any
got_request_exception: Any
appcontext_tearing_down: Any
appcontext_pushed: Any
appcontext_popped: Any
message_flashed: Any
