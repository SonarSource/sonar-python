from .._utils import sequence_to_str as sequence_to_str
from ._api import WeightsEnum as WeightsEnum
from torch import nn
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

class IntermediateLayerGetter(nn.ModuleDict):
    __annotations__: Any
    return_layers: Any
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None: ...
    def forward(self, x): ...
D = TypeVar('D')

def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]: ...
W = TypeVar('W', bound=WeightsEnum)
M = TypeVar('M', bound=nn.Module)
V = TypeVar('V')

def handle_legacy_interface(**weights: Tuple[str, Union[Optional[W], Callable[[Dict[str, Any]], Optional[W]]]]): ...

class _ModelURLs(dict):
    def __getitem__(self, item): ...
