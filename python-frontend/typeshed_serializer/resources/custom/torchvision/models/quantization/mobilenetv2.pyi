from .._api import WeightsEnum
from torch import Tensor
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNetV2, MobileNet_V2_Weights
from typing import Any, Optional, Union

class QuantizableInvertedResidual(InvertedResidual):
    skip_add: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class QuantizableMobileNetV2(MobileNetV2):
    quant: Any
    dequant: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class MobileNet_V2_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1: Any
    DEFAULT: Any

def mobilenet_v2(*, weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableMobileNetV2: ...