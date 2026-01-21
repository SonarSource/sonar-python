from typing import Any
from SonarPythonAnalyzerFakeStub import CustomStubBase

class TestClient(CustomStubBase):
    def __init__(
        self,
        app: Any,
        base_url: str = ...,
        raise_server_exceptions: bool = ...,
        root_path: str = ...,
        backend: str = ...,
        backend_options: dict[str, Any] | None = ...,
        cookies: Any = ...,
        headers: dict[str, str] | None = ...,
        follow_redirects: bool = ...,
        client: tuple[str, int] = ...,
    ) -> None: ...

    def request(
        self,
        method: str,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def get(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def options(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def head(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def post(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def put(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def patch(
        self,
        url: Any,
        *,
        content: Any = ...,
        data: Any = ...,
        files: Any = ...,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def delete(
        self,
        url: Any,
        *,
        params: Any = ...,
        headers: Any = ...,
        cookies: Any = ...,
        auth: Any = ...,
        follow_redirects: bool | Any = ...,
        timeout: Any = ...,
        extensions: dict[str, Any] | None = ...,
    ) -> Any: ...

    def websocket_connect(
        self,
        url: str,
        subprotocols: Any = ...,
        **kwargs: Any,
    ) -> Any: ...

    def __enter__(self) -> Any: ...
    def __exit__(self, *args: Any) -> None: ...

class WebSocketTestSession(CustomStubBase):
    def __enter__(self) -> Any: ...
    def __exit__(self, *args: Any) -> bool | None: ...
    def send(self, message: Any) -> None: ...
    def send_text(self, data: str) -> None: ...
    def send_bytes(self, data: bytes) -> None: ...
    def send_json(self, data: Any, mode: str = ...) -> None: ...
    def close(self, code: int = ..., reason: str | None = ...) -> None: ...
    def receive(self) -> Any: ...
    def receive_text(self) -> str: ...
    def receive_bytes(self) -> bytes: ...
    def receive_json(self, mode: str = ...) -> Any: ...
