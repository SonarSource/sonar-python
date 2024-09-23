from .._api import WeightsEnum
from ..mobilenetv3 import MobileNet_V3_Large_Weights
from .ssd import SSD, SSDScoringHead
from torch import Tensor, nn
from typing import Any, Callable, Dict, List, Optional

class SSDLiteHead(nn.Module):
    classification_head: Any
    regression_head: Any
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]) -> None: ...
    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]: ...

class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]) -> None: ...

class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: Callable[..., nn.Module]) -> None: ...

class SSDLiteFeatureExtractorMobileNet(nn.Module):
    features: Any
    extra: Any
    def __init__(self, backbone: nn.Module, c4_pos: int, norm_layer: Callable[..., nn.Module], width_mult: float = ..., min_depth: int = ...): ...
    def forward(self, x: Tensor) -> Dict[str, Tensor]: ...

class SSDLite320_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

def ssdlite320_mobilenet_v3_large(*, weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., **kwargs: Any) -> SSD: ...
