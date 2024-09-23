from .._api import WeightsEnum
from ..resnet import ResNet101_Weights, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from torch import nn
from typing import Any, Optional

class FCN(_SimpleSegmentationModel): ...

class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None: ...

class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1: Any
    DEFAULT: Any

class FCN_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1: Any
    DEFAULT: Any

def fcn_resnet50(*, weights: Optional[FCN_ResNet50_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., **kwargs: Any) -> FCN: ...
def fcn_resnet101(*, weights: Optional[FCN_ResNet101_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., aux_loss: Optional[bool] = ..., weights_backbone: Optional[ResNet101_Weights] = ..., **kwargs: Any) -> FCN: ...
