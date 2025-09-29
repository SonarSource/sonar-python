from typing import Union, List, Callable, Tuple, TypeVar, Any, overload

Tensor = TypeVar("Tensor")
ReductionCallable = Callable[[Tensor, Tuple[int, ...]], Tensor]
Reduction = Union[str, ReductionCallable]
Size = Any

@overload
def rearrange(tensor: List[Tensor], pattern: str, **axes_lengths: Size) -> Tensor: ...


@overload
def rearrange(tensor: Tensor, pattern: str, **axes_lengths: Size) -> Tensor: ...


def rearrange(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths: Size) -> Tensor: ...

@overload
def reduce(tensor: List[Tensor], pattern: str, reduction: Reduction, **axes_lengths: Size) -> Tensor: ...


@overload
def reduce(tensor: Tensor, pattern: str, reduction: Reduction, **axes_lengths: Size) -> Tensor: ...


def reduce(tensor: Union[Tensor, List[Tensor]], pattern: str, reduction: Reduction, **axes_lengths: Size) -> Tensor: ...

@overload
def repeat(tensor: List[Tensor], pattern: str, **axes_lengths: Size) -> Tensor: ...


@overload
def repeat(tensor: Tensor, pattern: str, **axes_lengths: Size) -> Tensor: ...


def repeat(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths: Size) -> Tensor: ...
