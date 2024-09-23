from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, Optional

class InvertedResidual(nn.Module):
    stride: Any
    use_res_connect: Any
    conv: Any
    out_channels: Any
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MobileNetV2(nn.Module):
    last_channel: Any
    features: Any
    classifier: Any
    def __init__(self, num_classes: int = ..., width_mult: float = ..., inverted_residual_setting: Optional[List[List[int]]] = ..., round_nearest: int = ..., block: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., dropout: float = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

def mobilenet_v2(*, weights: Optional[MobileNet_V2_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MobileNetV2: ...
