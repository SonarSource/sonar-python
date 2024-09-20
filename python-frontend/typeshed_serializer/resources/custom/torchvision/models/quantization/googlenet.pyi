from .._api import WeightsEnum
from ..googlenet import BasicConv2d, GoogLeNet, GoogLeNetOutputs, GoogLeNet_Weights, Inception, InceptionAux
from torch import Tensor
from typing import Any, Optional, Union

class QuantizableBasicConv2d(BasicConv2d):
    relu: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class QuantizableInception(Inception):
    cat: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class QuantizableInceptionAux(InceptionAux):
    relu: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class QuantizableGoogLeNet(GoogLeNet):
    quant: Any
    dequant: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> GoogLeNetOutputs: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class GoogLeNet_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

def googlenet(*, weights: Optional[Union[GoogLeNet_QuantizedWeights, GoogLeNet_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableGoogLeNet: ...