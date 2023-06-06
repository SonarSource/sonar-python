from typing import overload, TypeVar
from typing_extensions import TypeAlias

_ClassInfo: TypeAlias = type | tuple[_ClassInfo, ...]

class TestCase:
    def assertIsInstance(self, obj: object, cls: _ClassInfo) -> None: ...



class MyClassWithTypeVar:
    ...

_MyClassWithTypeVarT = TypeVar("_MyClassWithTypeVarT", bound=MyClassWithTypeVar)


@overload
def func_returning_bound_typevar() -> MyClassWithTypeVar:
    ...

@overload
def func_returning_bound_typevar() -> _MyClassWithTypeVarT:
    ...


_T = TypeVar("_T")

def func_returning_unbound_typevar(param: _T) -> _T:
    ...
