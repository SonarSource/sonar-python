import torch.nn as nn
import torch
from ._api import WeightsEnum
from typing import Any, Optional

class VGG(nn.Module):
    features: Any
    avgpool: Any
    classifier: Any
    def __init__(self, features: nn.Module, num_classes: int = ..., init_weights: bool = ..., dropout: float = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class VGG11_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class VGG11_BN_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class VGG13_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class VGG13_BN_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_FEATURES: Any
    DEFAULT: Any

class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class VGG19_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class VGG19_BN_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def vgg11(*, weights: Optional[VGG11_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg11_bn(*, weights: Optional[VGG11_BN_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg13(*, weights: Optional[VGG13_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg13_bn(*, weights: Optional[VGG13_BN_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg16(*, weights: Optional[VGG16_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg16_bn(*, weights: Optional[VGG16_BN_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg19(*, weights: Optional[VGG19_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
def vgg19_bn(*, weights: Optional[VGG19_BN_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VGG: ...
