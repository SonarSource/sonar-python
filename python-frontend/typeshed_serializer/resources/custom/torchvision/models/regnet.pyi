from ..ops.misc import Conv2dNormActivation
from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, Optional

class SimpleStemIN(Conv2dNormActivation):
    def __init__(self, width_in: int, width_out: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module]) -> None: ...

class BottleneckTransform(nn.Sequential):
    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]) -> None: ...

class ResBottleneckBlock(nn.Module):
    proj: Any
    f: Any
    activation: Any
    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int = ..., bottleneck_multiplier: float = ..., se_ratio: Optional[float] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class AnyStage(nn.Sequential):
    def __init__(self, width_in: int, width_out: int, stride: int, depth: int, block_constructor: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float] = ..., stage_index: int = ...) -> None: ...

class BlockParams:
    depths: Any
    widths: Any
    group_widths: Any
    bottleneck_multipliers: Any
    strides: Any
    se_ratio: Any
    def __init__(self, depths: List[int], widths: List[int], group_widths: List[int], bottleneck_multipliers: List[float], strides: List[int], se_ratio: Optional[float] = ...) -> None: ...
    @classmethod
    def from_init_params(cls, depth: int, w_0: int, w_a: float, w_m: float, group_width: int, bottleneck_multiplier: float = ..., se_ratio: Optional[float] = ..., **kwargs: Any) -> BlockParams: ...

class RegNet(nn.Module):
    stem: Any
    trunk_output: Any
    avgpool: Any
    fc: Any
    def __init__(self, block_params: BlockParams, num_classes: int = ..., stem_width: int = ..., stem_type: Optional[Callable[..., nn.Module]] = ..., block_type: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., activation: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class RegNet_Y_400MF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_Y_800MF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_Y_1_6GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_Y_3_2GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_Y_8GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_Y_16GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    IMAGENET1K_SWAG_E2E_V1: Any
    IMAGENET1K_SWAG_LINEAR_V1: Any
    DEFAULT: Any

class RegNet_Y_32GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    IMAGENET1K_SWAG_E2E_V1: Any
    IMAGENET1K_SWAG_LINEAR_V1: Any
    DEFAULT: Any

class RegNet_Y_128GF_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1: Any
    IMAGENET1K_SWAG_LINEAR_V1: Any
    DEFAULT: Any

class RegNet_X_400MF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_X_800MF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_X_1_6GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_X_3_2GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_X_8GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_X_16GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class RegNet_X_32GF_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

def regnet_y_400mf(*, weights: Optional[RegNet_Y_400MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_800mf(*, weights: Optional[RegNet_Y_800MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_1_6gf(*, weights: Optional[RegNet_Y_1_6GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_3_2gf(*, weights: Optional[RegNet_Y_3_2GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_8gf(*, weights: Optional[RegNet_Y_8GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_16gf(*, weights: Optional[RegNet_Y_16GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_32gf(*, weights: Optional[RegNet_Y_32GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_y_128gf(*, weights: Optional[RegNet_Y_128GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_400mf(*, weights: Optional[RegNet_X_400MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_800mf(*, weights: Optional[RegNet_X_800MF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_1_6gf(*, weights: Optional[RegNet_X_1_6GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_3_2gf(*, weights: Optional[RegNet_X_3_2GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_8gf(*, weights: Optional[RegNet_X_8GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_16gf(*, weights: Optional[RegNet_X_16GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
def regnet_x_32gf(*, weights: Optional[RegNet_X_32GF_Weights] = ..., progress: bool = ..., **kwargs: Any) -> RegNet: ...
