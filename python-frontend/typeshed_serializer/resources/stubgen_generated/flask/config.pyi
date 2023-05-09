import typing as t
from typing import Any

class ConfigAttribute:
    __name__: Any
    get_converter: Any
    def __init__(self, name: str, get_converter: t.Optional[t.Callable] = ...) -> None: ...
    def __get__(self, obj: t.Any, owner: t.Any = ...) -> t.Any: ...
    def __set__(self, obj: t.Any, value: t.Any) -> None: ...

class Config(dict):
    root_path: Any
    def __init__(self, root_path: str, defaults: t.Optional[dict] = ...) -> None: ...
    def from_envvar(self, variable_name: str, silent: bool = ...) -> bool: ...
    def from_prefixed_env(self, prefix: str = ..., *, loads: t.Callable[[str], t.Any] = ...) -> bool: ...
    def from_pyfile(self, filename: str, silent: bool = ...) -> bool: ...
    def from_object(self, obj: t.Union[object, str]) -> None: ...
    def from_file(self, filename: str, load: t.Callable[[t.IO[t.Any]], t.Mapping], silent: bool = ...) -> bool: ...
    def from_mapping(self, mapping: t.Optional[t.Mapping[str, t.Any]] = ..., **kwargs: t.Any) -> bool: ...
    def get_namespace(self, namespace: str, lowercase: bool = ..., trim_namespace: bool = ...) -> t.Dict[str, t.Any]: ...