import torch
from .._api import WeightsEnum
from ..mobilenetv3 import MobileNet_V3_Large_Weights
from ..resnet import ResNet101_Weights, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from torch import nn
from typing import Any, Optional, Sequence

class DeepLabV3(_SimpleSegmentationModel): ...

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = ...) -> None: ...

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None: ...

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ASPP(nn.Module):
    convs: Any
    project: Any
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1: Any
    DEFAULT: Any

class DeepLabV3_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1: Any
    DEFAULT: Any

class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1: Any
    DEFAULT: Any

def deeplabv3_resnet50(*, weights: Optional[DeepLabV3_ResNet50_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., **kwargs: Any) -> DeepLabV3: ...
def deeplabv3_resnet101(*, weights: Optional[DeepLabV3_ResNet101_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[ResNet101_Weights] = ..., **kwargs: Any) -> DeepLabV3: ...
def deeplabv3_mobilenet_v3_large(*, weights: Optional[DeepLabV3_MobileNet_V3_Large_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., **kwargs: Any) -> DeepLabV3: ...
