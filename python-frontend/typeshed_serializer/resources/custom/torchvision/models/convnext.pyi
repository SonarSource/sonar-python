from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, Optional
from SonarPythonAnalyzerFakeStub import CustomStubBase

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor: ...

class CNBlock(nn.Module):
    block: Any
    layer_scale: Any
    stochastic_depth: Any
    def __init__(self, dim, layer_scale: float, stochastic_depth_prob: float, norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class CNBlockConfig(CustomStubBase):
    input_channels: Any
    out_channels: Any
    num_layers: Any
    def __init__(self, input_channels: int, out_channels: Optional[int], num_layers: int) -> None: ...

class ConvNeXt(nn.Module):
    features: Any
    avgpool: Any
    classifier: Any
    def __init__(self, block_setting: List[CNBlockConfig], stochastic_depth_prob: float = ..., layer_scale: float = ..., num_classes: int = ..., block: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ConvNeXt_Small_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ConvNeXt_Large_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ConvNeXt: ...
def convnext_small(*, weights: Optional[ConvNeXt_Small_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ConvNeXt: ...
def convnext_base(*, weights: Optional[ConvNeXt_Base_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ConvNeXt: ...
def convnext_large(*, weights: Optional[ConvNeXt_Large_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ConvNeXt: ...
