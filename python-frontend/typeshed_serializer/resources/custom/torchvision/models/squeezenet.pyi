import torch.nn as nn
import torch
from ._api import WeightsEnum
from typing import Any, Optional

class Fire(nn.Module):
    inplanes: Any
    squeeze: Any
    squeeze_activation: Any
    expand1x1: Any
    expand1x1_activation: Any
    expand3x3: Any
    expand3x3_activation: Any
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class SqueezeNet(nn.Module):
    num_classes: Any
    features: Any
    classifier: Any
    def __init__(self, version: str = ..., num_classes: int = ..., dropout: float = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class SqueezeNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class SqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def squeezenet1_0(*, weights: Optional[SqueezeNet1_0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SqueezeNet: ...
def squeezenet1_1(*, weights: Optional[SqueezeNet1_1_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SqueezeNet: ...
