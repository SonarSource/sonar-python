from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, Optional, Sequence, Union

class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]
    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = ...) -> int: ...
    def __init__(self, expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block) -> None: ...

class MBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int, out_channels: int, num_layers: int, width_mult: float = ..., depth_mult: float = ..., block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float): ...

class FusedMBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int, out_channels: int, num_layers: int, block: Optional[Callable[..., nn.Module]] = ...) -> None: ...

class MBConv(nn.Module):
    use_res_connect: Any
    block: Any
    stochastic_depth: Any
    out_channels: Any
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module], se_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class FusedMBConv(nn.Module):
    use_res_connect: Any
    block: Any
    stochastic_depth: Any
    out_channels: Any
    def __init__(self, cnf: FusedMBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module]) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class EfficientNet(nn.Module):
    features: Any
    avgpool: Any
    classifier: Any
    def __init__(self, inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]], dropout: float, stochastic_depth_prob: float = ..., num_classes: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., last_channel: Optional[int] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class EfficientNet_B0_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_B1_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class EfficientNet_B2_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_B3_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_B4_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_B5_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_B6_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_B7_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_V2_M_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class EfficientNet_V2_L_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def efficientnet_b0(*, weights: Optional[EfficientNet_B0_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b1(*, weights: Optional[EfficientNet_B1_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b2(*, weights: Optional[EfficientNet_B2_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b3(*, weights: Optional[EfficientNet_B3_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b4(*, weights: Optional[EfficientNet_B4_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b5(*, weights: Optional[EfficientNet_B5_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b6(*, weights: Optional[EfficientNet_B6_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_b7(*, weights: Optional[EfficientNet_B7_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_v2_s(*, weights: Optional[EfficientNet_V2_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_v2_m(*, weights: Optional[EfficientNet_V2_M_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
def efficientnet_v2_l(*, weights: Optional[EfficientNet_V2_L_Weights] = ..., progress: bool = ..., **kwargs: Any) -> EfficientNet: ...
