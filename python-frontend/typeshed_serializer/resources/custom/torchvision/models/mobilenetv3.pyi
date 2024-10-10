from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, Optional
from SonarPythonAnalyzerFakeStub import CustomStubBase

class InvertedResidualConfig(CustomStubBase):
    input_channels: Any
    kernel: Any
    expanded_channels: Any
    out_channels: Any
    use_se: Any
    use_hs: Any
    stride: Any
    dilation: Any
    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool, activation: str, stride: int, dilation: int, width_mult: float) -> None: ...
    @staticmethod
    def adjust_channels(channels: int, width_mult: float): ...

class InvertedResidual(nn.Module):
    use_res_connect: Any
    block: Any
    out_channels: Any
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module], se_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class MobileNetV3(nn.Module):
    features: Any
    avgpool: Any
    classifier: Any
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, num_classes: int = ..., block: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., dropout: float = ..., **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MobileNet_V3_Large_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_V2: Any
    DEFAULT: Any

class MobileNet_V3_Small_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def mobilenet_v3_large(*, weights: Optional[MobileNet_V3_Large_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MobileNetV3: ...
def mobilenet_v3_small(*, weights: Optional[MobileNet_V3_Small_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MobileNetV3: ...
