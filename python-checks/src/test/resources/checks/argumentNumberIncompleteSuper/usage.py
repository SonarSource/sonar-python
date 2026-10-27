from base import FiniteField
from decorators import public


class GMPYIntegerRing:
    pass


class PythonIntegerRing:
    pass


@public
class GMPYFiniteField(FiniteField):
    def __init__(self, mod, symmetric=True):
        super().__init__(mod, GMPYIntegerRing(), symmetric)  # Noncompliant {{Remove 1 unexpected arguments; '__init__' expects at most 2 positional arguments.}}
        super().__init__(mod, symmetric)
        super().__init__(mod, symmetric=symmetric)


@public
class PythonFiniteField(FiniteField):
    def __init__(self, mod, symmetric=True):
        super().__init__(mod, PythonIntegerRing(), symmetric)  # Noncompliant
        super().__init__(mod, symmetric)
        super().__init__(mod, symmetric=symmetric)
