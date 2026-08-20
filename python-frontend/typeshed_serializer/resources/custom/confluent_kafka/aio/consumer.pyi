from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any, Callable, Coroutine, List, Optional

class AIOConsumer(CustomStubBase):
    def __init__(self, config: dict, **kwargs: Any) -> None: ...
    async def subscribe(
        self,
        topics: List[str],
        on_assign: Optional[Callable[..., Coroutine[Any, Any, None]]] = ...,
        on_revoke: Optional[Callable[..., Coroutine[Any, Any, None]]] = ...,
        on_lost: Optional[Callable[..., Coroutine[Any, Any, None]]] = ...,
        **kwargs: Any,
    ) -> None: ...
