from typing import Any, Callable, Optional

# URL type (FQN: pydantic_core._pydantic_core.Url, exposed as pydantic_core.Url)
# Identical interface to pydantic.networks.AnyUrl/_BaseUrl — same object in pydantic v2.
class Url:
    scheme: str
    host: str
    path: str
    port: int
    query: Optional[str]
    fragment: Optional[str]
    username: Optional[str]
    password: Optional[str]
    def unicode_string(self) -> str: ...
    def unicode_host(self) -> str: ...
    def query_params(self) -> list: ...

def to_json(
    value: Any,
    *,
    indent: Optional[int] = ...,
    ensure_ascii: bool = ...,
    include: Any = ...,
    exclude: Any = ...,
    by_alias: bool = ...,
    exclude_none: bool = ...,
    round_trip: bool = ...,
    timedelta_mode: str = ...,
    temporal_mode: str = ...,
    bytes_mode: str = ...,
    inf_nan_mode: str = ...,
    serialize_unknown: bool = ...,
    fallback: Optional[Callable[..., Any]] = ...,
    serialize_as_any: bool = ...,
    polymorphic_serialization: Optional[bool] = ...,
    context: Optional[Any] = ...,
) -> bytes: ...

def to_jsonable_python(
    value: Any,
    *,
    include: Any = ...,
    exclude: Any = ...,
    by_alias: bool = ...,
    exclude_none: bool = ...,
    round_trip: bool = ...,
    timedelta_mode: str = ...,
    temporal_mode: str = ...,
    bytes_mode: str = ...,
    inf_nan_mode: str = ...,
    serialize_unknown: bool = ...,
    fallback: Optional[Callable[..., Any]] = ...,
    serialize_as_any: bool = ...,
    polymorphic_serialization: Optional[bool] = ...,
    context: Optional[Any] = ...,
) -> Any: ...
