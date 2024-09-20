from .._api import WeightsEnum
from ..vgg import VGG16_Weights
from .anchor_utils import DefaultBoxGenerator
from torch import Tensor, nn
from typing import Any, Dict, List, Optional, Tuple

class SSD300_VGG16_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

class SSDHead(nn.Module):
    classification_head: Any
    regression_head: Any
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int) -> None: ...
    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]: ...

class SSDScoringHead(nn.Module):
    module_list: Any
    num_columns: Any
    def __init__(self, module_list: nn.ModuleList, num_columns: int) -> None: ...
    def forward(self, x: List[Tensor]) -> Tensor: ...

class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int) -> None: ...

class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]) -> None: ...

class SSD(nn.Module):
    __annotations__: Any
    backbone: Any
    anchor_generator: Any
    box_coder: Any
    head: Any
    proposal_matcher: Any
    transform: Any
    score_thresh: Any
    nms_thresh: Any
    detections_per_img: Any
    topk_candidates: Any
    neg_to_pos_ratio: Any
    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator, size: Tuple[int, int], num_classes: int, image_mean: Optional[List[float]] = ..., image_std: Optional[List[float]] = ..., head: Optional[nn.Module] = ..., score_thresh: float = ..., nms_thresh: float = ..., detections_per_img: int = ..., iou_thresh: float = ..., topk_candidates: int = ..., positive_fraction: float = ..., **kwargs: Any) -> None: ...
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]: ...
    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = ...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...
    def postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]: ...

class SSDFeatureExtractorVGG(nn.Module):
    scale_weight: Any
    features: Any
    extra: Any
    def __init__(self, backbone: nn.Module, highres: bool) -> None: ...
    def forward(self, x: Tensor) -> Dict[str, Tensor]: ...

def ssd300_vgg16(*, weights: Optional[SSD300_VGG16_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[VGG16_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> SSD: ...
