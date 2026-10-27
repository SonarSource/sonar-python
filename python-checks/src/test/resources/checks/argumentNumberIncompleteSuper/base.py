from typing import Generic, TypeVar
from decorators import public, doctest_depends_on

T = TypeVar("T")


class Domain(Generic[T]):
    pass


class Field(Domain[T]):
    pass


class SimpleDomain(Domain[T]):
    pass


# The SymPy shape: a decorated direct parent with distant generic ancestry.
@public
@doctest_depends_on(modules=["python", "gmpy"])
class FiniteField(Field, SimpleDomain):
    def __init__(self, mod, symmetric=True):
        pass
