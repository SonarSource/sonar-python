from SonarPythonAnalyzerFakeStub import CustomStubBase
import torch.nn as nn

from typing import Any, IO, Optional, Sequence, Union

class dtype(CustomStubBase): ...
class layout(CustomStubBase): ...
class device(CustomStubBase):
    def __init__(
        self,
        type: str,
        index: Optional[int] = ...,
    ) -> None: ...

class Tensor(CustomStubBase):
    def new_tensor(
        self,
        data: Any,
        dtype: Optional[dtype] = ...,
        device: Optional[Union[str, device]] = ...,
        requires_grad: Optional[bool] = ...,
        pin_memory: bool = ...,
    ) -> "Tensor": ...

    def flatten(self, start_dim: int = ..., end_dim: int = ...) -> Tensor: ...

# --- Common Tensor Creation Functions ---

def tensor(
    data: Any,
    dtype: Optional[dtype] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...


def empty(
    *size: int,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def zeros(
    *size: int,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def ones(
    *size: int,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def rand(
    *size: int,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def randn(
    *size: int,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def arange(
    start: Union[int, float],
    end: Optional[Union[int, float]] = ...,
    step: Union[int, float] = ...,
    *,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def full(
    size: Union[int, Sequence[int]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

def eye(
    n: int,
    m: Optional[int] = ...,
    *,
    dtype: Optional[dtype] = ...,
    layout: Optional[layout] = ...,
    device: Optional[Union[str, device]] = ...,
    requires_grad: bool = ...,
    pin_memory: bool = ...,
) -> Tensor: ...

# --- Serialization Functions ---

def save(
    obj: Any,
    f: Union[str, IO[bytes]],
    pickle_module: Any = ...,
    pickle_protocol: int = ...,
    _use_new_zipfile_serialization: bool = ...,
) -> None: ...

def load(
    f: Union[str, IO[bytes]],
    map_location: Optional[Union[str, device]] = ...,
    pickle_module: Any = ...,
    **pickle_load_args: Any
) -> Any: ...

# --- Mathematical Functions ---

def log(
    input: Tensor,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor: ...

def exp(
    input: Tensor,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor: ...

def log1p(
    input: Tensor,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor: ...

def expm1(
    input: Tensor,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor: ...

# --- Tensor manipulation ---

def flatten(intput:Tensor, start_dim: int = ..., end_dim: int = ...) -> Tensor: ...
