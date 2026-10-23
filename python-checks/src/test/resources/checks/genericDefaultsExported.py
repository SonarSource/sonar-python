from anyio.abc import TaskStatus
from typing import Generic, TypeVar

T = TypeVar("T")

class Required(Generic[T]):
    pass
