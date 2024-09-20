import torch.nn as nn
from ._api import WeightsEnum
from torch import Tensor
from typing import Any, List, Optional, Tuple

class _DenseLayer(nn.Module):
    norm1: Any
    relu1: Any
    conv1: Any
    norm2: Any
    relu2: Any
    conv2: Any
    drop_rate: Any
    memory_efficient: Any
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = ...) -> None: ...
    def bn_function(self, inputs: List[Tensor]) -> Tensor: ...
    def any_requires_grad(self, input: List[Tensor]) -> bool: ...
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor: ...

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float, memory_efficient: bool = ...) -> None: ...
    def forward(self, init_features: Tensor) -> Tensor: ...

class _Transition(nn.Sequential):
    norm: Any
    relu: Any
    conv: Any
    pool: Any
    def __init__(self, num_input_features: int, num_output_features: int) -> None: ...

class DenseNet(nn.Module):
    features: Any
    classifier: Any
    def __init__(self, growth_rate: int = ..., block_config: Tuple[int, int, int, int] = ..., num_init_features: int = ..., bn_size: int = ..., drop_rate: float = ..., num_classes: int = ..., memory_efficient: bool = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class DenseNet121_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class DenseNet161_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class DenseNet169_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class DenseNet201_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def densenet121(*, weights: Optional[DenseNet121_Weights] = ..., progress: bool = ..., **kwargs: Any) -> DenseNet: ...
def densenet161(*, weights: Optional[DenseNet161_Weights] = ..., progress: bool = ..., **kwargs: Any) -> DenseNet: ...
def densenet169(*, weights: Optional[DenseNet169_Weights] = ..., progress: bool = ..., **kwargs: Any) -> DenseNet: ...
def densenet201(*, weights: Optional[DenseNet201_Weights] = ..., progress: bool = ..., **kwargs: Any) -> DenseNet: ...
