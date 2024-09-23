import torch.nn as nn
import torch
from ._api import WeightsEnum
from torch import Tensor
from typing import Any, Optional

class _InvertedResidual(nn.Module):
    apply_residual: Any
    layers: Any
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, expansion_factor: int, bn_momentum: float = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class MNASNet(torch.nn.Module):
    alpha: Any
    num_classes: Any
    layers: Any
    classifier: Any
    def __init__(self, alpha: float, num_classes: int = ..., dropout: float = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MNASNet0_5_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class MNASNet0_75_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class MNASNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class MNASNet1_3_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def mnasnet0_5(*, weights: Optional[MNASNet0_5_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MNASNet: ...
def mnasnet0_75(*, weights: Optional[MNASNet0_75_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MNASNet: ...
def mnasnet1_0(*, weights: Optional[MNASNet1_0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MNASNet: ...
def mnasnet1_3(*, weights: Optional[MNASNet1_3_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MNASNet: ...
