from SonarPythonAnalyzerFakeStub import CustomStubBase
import torch.nn as nn

from typing import Any, Optional, Sequence, Union

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
