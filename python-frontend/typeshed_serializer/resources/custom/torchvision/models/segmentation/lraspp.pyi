from .._api import WeightsEnum
from ..mobilenetv3 import MobileNet_V3_Large_Weights
from torch import Tensor, nn
from typing import Any, Dict, Optional

class LRASPP(nn.Module):
    backbone: Any
    classifier: Any
    def __init__(self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = ...) -> None: ...
    def forward(self, input: Tensor) -> Dict[str, Tensor]: ...

class LRASPPHead(nn.Module):
    cbr: Any
    scale: Any
    low_classifier: Any
    high_classifier: Any
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None: ...
    def forward(self, input: Dict[str, Tensor]) -> Tensor: ...

class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1: Any
    DEFAULT: Any

def lraspp_mobilenet_v3_large(*, weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., **kwargs: Any) -> LRASPP: ...
