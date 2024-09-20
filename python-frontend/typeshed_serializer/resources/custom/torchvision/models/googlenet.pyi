import torch.nn as nn
from ._api import WeightsEnum
from torch import Tensor
from typing import Any, Callable, List, NamedTuple, Optional

class GoogLeNetOutputs(NamedTuple):
    logits: Any
    aux_logits2: Any
    aux_logits1: Any

class GoogLeNet(nn.Module):
    __constants__: Any
    aux_logits: Any
    transform_input: Any
    conv1: Any
    maxpool1: Any
    conv2: Any
    conv3: Any
    maxpool2: Any
    inception3a: Any
    inception3b: Any
    maxpool3: Any
    inception4a: Any
    inception4b: Any
    inception4c: Any
    inception4d: Any
    inception4e: Any
    maxpool4: Any
    inception5a: Any
    inception5b: Any
    aux1: Any
    aux2: Any
    avgpool: Any
    dropout: Any
    fc: Any
    def __init__(self, num_classes: int = ..., aux_logits: bool = ..., transform_input: bool = ..., init_weights: Optional[bool] = ..., blocks: Optional[List[Callable[..., nn.Module]]] = ..., dropout: float = ..., dropout_aux: float = ...) -> None: ...
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs: ...
    def forward(self, x: Tensor) -> GoogLeNetOutputs: ...

class Inception(nn.Module):
    branch1: Any
    branch2: Any
    branch3: Any
    branch4: Any
    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, ch5x5red: int, ch5x5: int, pool_proj: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class InceptionAux(nn.Module):
    conv: Any
    fc1: Any
    fc2: Any
    dropout: Any
    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = ..., dropout: float = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class BasicConv2d(nn.Module):
    conv: Any
    bn: Any
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class GoogLeNet_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def googlenet(*, weights: Optional[GoogLeNet_Weights] = ..., progress: bool = ..., **kwargs: Any) -> GoogLeNet: ...

# Names in __all__ with no definition:
#   GoogLeNetOutputs
#   _GoogLeNetOutputs
