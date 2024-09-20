import torch.nn as nn
from .._api import WeightsEnum
from torch import Tensor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes: int, out_planes: int, midplanes: Optional[int] = ..., stride: int = ..., padding: int = ...) -> None: ...
    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]: ...

class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = ..., padding: int = ...) -> None: ...
    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]: ...

class Conv3DNoTemporal(nn.Conv3d):
    def __init__(self, in_planes: int, out_planes: int, midplanes: Optional[int] = ..., stride: int = ..., padding: int = ...) -> None: ...
    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]: ...

class BasicBlock(nn.Module):
    expansion: int
    conv1: Any
    conv2: Any
    relu: Any
    downsample: Any
    stride: Any
    def __init__(self, inplanes: int, planes: int, conv_builder: Callable[..., nn.Module], stride: int = ..., downsample: Optional[nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Bottleneck(nn.Module):
    expansion: int
    conv1: Any
    conv2: Any
    conv3: Any
    relu: Any
    downsample: Any
    stride: Any
    def __init__(self, inplanes: int, planes: int, conv_builder: Callable[..., nn.Module], stride: int = ..., downsample: Optional[nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class BasicStem(nn.Sequential):
    def __init__(self) -> None: ...

class R2Plus1dStem(nn.Sequential):
    def __init__(self) -> None: ...

class VideoResNet(nn.Module):
    inplanes: int
    stem: Any
    layer1: Any
    layer2: Any
    layer3: Any
    layer4: Any
    avgpool: Any
    fc: Any
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], conv_makers: Sequence[Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]]], layers: List[int], stem: Callable[..., nn.Module], num_classes: int = ..., zero_init_residual: bool = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class R3D_18_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

class MC3_18_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

class R2Plus1D_18_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

def r3d_18(*, weights: Optional[R3D_18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VideoResNet: ...
def mc3_18(*, weights: Optional[MC3_18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VideoResNet: ...
def r2plus1d_18(*, weights: Optional[R2Plus1D_18_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VideoResNet: ...
