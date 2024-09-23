from .._api import WeightsEnum
from ..shufflenetv2 import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights
from torch import Tensor
from torchvision.models import shufflenetv2
from typing import Any, Optional, Union

class QuantizableInvertedResidual(shufflenetv2.InvertedResidual):
    cat: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class QuantizableShuffleNetV2(shufflenetv2.ShuffleNetV2):
    quant: Any
    dequant: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class ShuffleNet_V2_X0_5_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

class ShuffleNet_V2_X1_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

class ShuffleNet_V2_X1_5_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

class ShuffleNet_V2_X2_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

def shufflenet_v2_x0_5(*, weights: Optional[Union[ShuffleNet_V2_X0_5_QuantizedWeights, ShuffleNet_V2_X0_5_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableShuffleNetV2: ...
def shufflenet_v2_x1_0(*, weights: Optional[Union[ShuffleNet_V2_X1_0_QuantizedWeights, ShuffleNet_V2_X1_0_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableShuffleNetV2: ...
def shufflenet_v2_x1_5(*, weights: Optional[Union[ShuffleNet_V2_X1_5_QuantizedWeights, ShuffleNet_V2_X1_5_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableShuffleNetV2: ...
def shufflenet_v2_x2_0(*, weights: Optional[Union[ShuffleNet_V2_X2_0_QuantizedWeights, ShuffleNet_V2_X2_0_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableShuffleNetV2: ...
