from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any, Dict, List, Callable

class AsyncClient(CustomStubBase):
    def __init__(self, *, event_hooks: Dict[str, List[Callable[..., Any]]] = ..., **kwargs: Any) -> None: ...

class Client(CustomStubBase):
    def __init__(self, *, event_hooks: Dict[str, List[Callable[..., Any]]] = ..., **kwargs: Any) -> None: ...
