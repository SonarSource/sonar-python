def non_compliant(xx):
    from typing import TypeVar

    _T = TypeVar("_T", bound=str)
    _R = TypeVar("_R")
    _S = TypeVar("_S")

    def func(a: _T, b: int) -> str:  # Noncompliant {{Use a generic type parameter for this function instead of a TypeVar.}}
        #       ^^
        ...
    def func(a: _T, b: int) -> _T:  # Noncompliant 2
        ...

    def func(a: _T, b: _R) -> _S:  # Noncompliant 3
        ...


def compliant(xx):
    def func[T: str](a: T, b: int) -> T: # OK.
        ...

    def func[T](a: R, b: int) -> tuple(T,R): # OK.
        ...
