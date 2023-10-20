
from typing import TypeVar

def non_compliant_1():
    _T = TypeVar("_T", bound=str)
    #    ^^^^^^^^^^^^^^^^^^^^^^^^>  {{"TypeVar" is assigned here.}}
    def func(a: _T, b: int) -> str:  # Noncompliant {{Use a generic type parameter for this function instead of a "TypeVar".}}
    #   ^^^^    ^^<                                 {{Use of "TypeVar" here.}}
        ...
def non_compliant_2():
    _T = TypeVar("_T", bound=str)
    #    ^^^^^^^^^^^^^^^^^^^^^^^^>     {{"TypeVar" is assigned here.}}

    def func(a: _T, b: int) -> _T:  # Noncompliant {{Use a generic type parameter for this function instead of a "TypeVar".}}
    #   ^^^^    ^^<                                {{Use of "TypeVar" here.}}
    #                          ^^@-1<              {{Use of "TypeVar" here.}}
        ...

def non_compliant_3():
    _T = TypeVar("_T", bound=str)
    #    ^^^^^^^^^^^^^^^^^^^^^^^^>  {{"TypeVar" is assigned here.}}
    _R = TypeVar("_R")
    #    ^^^^^^^^^^^^^>             {{"TypeVar" is assigned here.}}
    _S = TypeVar("_S")
    #    ^^^^^^^^^^^^^>             {{"TypeVar" is assigned here.}}
    def func(a: _T, b: _R) -> _S:  # Noncompliant {{Use a generic type parameter for this function instead of a "TypeVar".}}
    #   ^^^^    ^^<                               {{Use of "TypeVar" here.}}
    #                  ^^@-1<                     {{Use of "TypeVar" here.}}
    #                         ^^@-2<              {{Use of "TypeVar" here.}}
        ...


def FN_non_compliant():
    _T = TypeVar("_T", bound=str)

    def func(a: list[_T]) -> set[_T]:  # FN
        ...

def compliant_1[T: str](a: T, b: int) -> T: # OK.
    ...

def compliant_2[T,R](a: R, b: int) -> tuple(T,R): # OK.
    ...
