from typing import Any as _Any

class Any:
    my_var: Any

    no_hint_var = "test"

    def __new__(cls) -> Any:
        return object.__new__(cls)


def foo(test: str, param: Any) -> str:
    pass

def foobar() -> Any:
    pass

def proper_any(param: _Any) -> _Any: # Noncompliant 2
    my_normal_any_var: _Any # Noncompliant {{Use a more specific type than `Any` for this type hint.}}
                      #^^^^
    pass

def success_without_hint(param):
    pass
