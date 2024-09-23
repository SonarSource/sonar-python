import torch.nn as nn
from ._api import WeightsEnum
from torch import Tensor
from typing import Any, Callable, List, Optional

class InvertedResidual(nn.Module):
    stride: Any
    branch1: Any
    branch2: Any
    def __init__(self, inp: int, oup: int, stride: int) -> None: ...
    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = ..., padding: int = ..., bias: bool = ...) -> nn.Conv2d: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ShuffleNetV2(nn.Module):
    conv1: Any
    maxpool: Any
    stage2: Any
    stage3: Any
    stage4: Any
    conv5: Any
    fc: Any
    def __init__(self, stages_repeats: List[int], stages_out_channels: List[int], num_classes: int = ..., inverted_residual: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ShuffleNet_V2_X0_5_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ShuffleNet_V2_X1_0_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ShuffleNet_V2_X1_5_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ShuffleNet_V2_X2_0_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def shufflenet_v2_x0_5(*, weights: Optional[ShuffleNet_V2_X0_5_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ShuffleNetV2: ...
def shufflenet_v2_x1_0(*, weights: Optional[ShuffleNet_V2_X1_0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ShuffleNetV2: ...
def shufflenet_v2_x1_5(*, weights: Optional[ShuffleNet_V2_X1_5_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ShuffleNetV2: ...
def shufflenet_v2_x2_0(*, weights: Optional[ShuffleNet_V2_X2_0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> ShuffleNetV2: ...
