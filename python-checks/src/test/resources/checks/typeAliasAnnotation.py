def scope_1():
    from typing import TypeAlias, TypeVar

    _T = TypeVar("_T")

    BadTypeAlias: TypeAlias = set[_T]  # Noncompliant {{Use a "type" statement instead of this "TypeAlias".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    type GoodTypeAlias[T] = set[T] # OK

def scope_2():
    import typing as tp

    _T = tp.TypeVar("_T")

    BadTypeAlias: tp.TypeAlias = set[_T]  # Noncompliant {{Use a "type" statement instead of this "TypeAlias".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def scope_3():
    from typing import TypeVar
    _T = TypeVar("_T")
    class TypeAlias:
        ...

    BadTypeAlias: TypeAlias = set[_T]


