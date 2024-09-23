import torch.nn as nn
from ._api import WeightsEnum
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union

class BasicBlock(nn.Module):
    expansion: int
    conv1: Any
    bn1: Any
    relu: Any
    conv2: Any
    bn2: Any
    downsample: Any
    stride: Any
    def __init__(self, inplanes: int, planes: int, stride: int = ..., downsample: Optional[nn.Module] = ..., groups: int = ..., base_width: int = ..., dilation: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Bottleneck(nn.Module):
    expansion: int
    conv1: Any
    bn1: Any
    conv2: Any
    bn2: Any
    conv3: Any
    bn3: Any
    relu: Any
    downsample: Any
    stride: Any
    def __init__(self, inplanes: int, planes: int, stride: int = ..., downsample: Optional[nn.Module] = ..., groups: int = ..., base_width: int = ..., dilation: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ResNet(nn.Module):
    inplanes: int
    dilation: int
    groups: Any
    base_width: Any
    conv1: Any
    bn1: Any
    relu: Any
    maxpool: Any
    layer1: Any
    layer2: Any
    layer3: Any
    layer4: Any
    avgpool: Any
    fc: Any
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = ..., zero_init_residual: bool = ..., groups: int = ..., width_per_group: int = ..., replace_stride_with_dilation: Optional[List[bool]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ResNet34_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class ResNet101_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class ResNet152_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class ResNeXt50_32X4D_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class ResNeXt101_32X8D_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class ResNeXt101_64X4D_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class Wide_ResNet101_2_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

def resnet18(*, weights: Optional[ResNet18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnet34(*, weights: Optional[ResNet34_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnet50(*, weights: Optional[ResNet50_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnet101(*, weights: Optional[ResNet101_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnet152(*, weights: Optional[ResNet152_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnext50_32x4d(*, weights: Optional[ResNeXt50_32X4D_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnext101_32x8d(*, weights: Optional[ResNeXt101_32X8D_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def resnext101_64x4d(*, weights: Optional[ResNeXt101_64X4D_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def wide_resnet50_2(*, weights: Optional[Wide_ResNet50_2_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
def wide_resnet101_2(*, weights: Optional[Wide_ResNet101_2_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ResNet: ...
