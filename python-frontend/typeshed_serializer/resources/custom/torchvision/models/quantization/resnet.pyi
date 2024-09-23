from .._api import WeightsEnum
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights, ResNet, ResNet18_Weights, ResNet50_Weights
from typing import Any, Optional, Union

class QuantizableBasicBlock(BasicBlock):
    add_relu: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class QuantizableBottleneck(Bottleneck):
    skip_add_relu: Any
    relu1: Any
    relu2: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class QuantizableResNet(ResNet):
    quant: Any
    dequant: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def fuse_model(self, is_qat: Optional[bool] = ...) -> None: ...

class ResNet18_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

class ResNet50_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    IMAGENET1K_FBGEMM_V2: Any
    DEFAULT: Any

class ResNeXt101_32X8D_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    IMAGENET1K_FBGEMM_V2: Any
    DEFAULT: Any

class ResNeXt101_64X4D_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1: Any
    DEFAULT: Any

def resnet18(*, weights: Optional[Union[ResNet18_QuantizedWeights, ResNet18_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableResNet: ...
def resnet50(*, weights: Optional[Union[ResNet50_QuantizedWeights, ResNet50_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableResNet: ...
def resnext101_32x8d(*, weights: Optional[Union[ResNeXt101_32X8D_QuantizedWeights, ResNeXt101_32X8D_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableResNet: ...
def resnext101_64x4d(*, weights: Optional[Union[ResNeXt101_64X4D_QuantizedWeights, ResNeXt101_64X4D_Weights]] = ..., progress: bool = ..., quantize: bool = ..., **kwargs: Any) -> QuantizableResNet: ...
