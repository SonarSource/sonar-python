from typing import Any, Callable, Sequence, TypeVar, Union

_T = TypeVar("_T")

def xfail(*args: Any, reason: str = ..., **kwargs: Any) -> Callable[[_T], _T]: ...
def skip(*args: Any, reason: str = ..., **kwargs: Any) -> Callable[[_T], _T]: ...
def parametrize(
    argnames: Union[str, Sequence[str]],
    argvalues: object,
    *args: Any,
    **kwargs: Any,
) -> Callable[[_T], _T]: ...
