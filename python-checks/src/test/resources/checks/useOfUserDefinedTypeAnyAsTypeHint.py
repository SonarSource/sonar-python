from typing import Any as _Any

class Any:
    def __new__(cls) -> Any:
        return object.__new__(cls)

def foo(test: str, param: Any) -> str:
    pass

def foobar() -> Any:
    pass

def proper_any(param: _Any) -> _Any: # Noncompliant 2
    pass
