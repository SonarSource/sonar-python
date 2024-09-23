import torch
from .._api import WeightsEnum
from torch import nn
from typing import Any, Callable, Optional

class TemporalSeparableConv(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int, padding: int, norm_layer: Callable[..., nn.Module]) -> None: ...

class SepInceptionBlock3D(nn.Module):
    branch0: Any
    branch1: Any
    branch2: Any
    branch3: Any
    def __init__(self, in_planes: int, b0_out: int, b1_mid: int, b1_out: int, b2_mid: int, b2_out: int, b3_out: int, norm_layer: Callable[..., nn.Module]) -> None: ...
    def forward(self, x): ...

class S3D(nn.Module):
    features: Any
    avgpool: Any
    classifier: Any
    def __init__(self, num_classes: int = ..., dropout: float = ..., norm_layer: Optional[Callable[..., torch.nn.Module]] = ...) -> None: ...
    def forward(self, x): ...

class S3D_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

def s3d(*, weights: Optional[S3D_Weights] = ..., progress: bool = ..., **kwargs: Any) -> S3D: ...
