from .._api import WeightsEnum
from ..resnet import ResNet50_Weights
from torch import Tensor, nn
from typing import Any, Callable, Dict, List, Optional, Tuple

class RetinaNetHead(nn.Module):
    classification_head: Any
    regression_head: Any
    def __init__(self, in_channels, num_anchors, num_classes, norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]: ...
    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]: ...

class RetinaNetClassificationHead(nn.Module):
    conv: Any
    cls_logits: Any
    num_classes: Any
    num_anchors: Any
    BETWEEN_THRESHOLDS: Any
    def __init__(self, in_channels, num_anchors, num_classes, prior_probability: float = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], matched_idxs: List[Tensor]) -> Tensor: ...
    def forward(self, x: List[Tensor]) -> Tensor: ...

class RetinaNetRegressionHead(nn.Module):
    __annotations__: Any
    conv: Any
    bbox_reg: Any
    box_coder: Any
    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Tensor: ...
    def forward(self, x: List[Tensor]) -> Tensor: ...

class RetinaNet(nn.Module):
    __annotations__: Any
    backbone: Any
    anchor_generator: Any
    head: Any
    proposal_matcher: Any
    box_coder: Any
    transform: Any
    score_thresh: Any
    nms_thresh: Any
    detections_per_img: Any
    topk_candidates: Any
    def __init__(self, backbone, num_classes, min_size: int = ..., max_size: int = ..., image_mean: Any | None = ..., image_std: Any | None = ..., anchor_generator: Any | None = ..., head: Any | None = ..., proposal_matcher: Any | None = ..., score_thresh: float = ..., nms_thresh: float = ..., detections_per_img: int = ..., fg_iou_thresh: float = ..., bg_iou_thresh: float = ..., topk_candidates: int = ..., **kwargs) -> None: ...
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor]) -> Dict[str, Tensor]: ...
    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]: ...
    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = ...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...

class RetinaNet_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

class RetinaNet_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

def retinanet_resnet50_fpn(*, weights: Optional[RetinaNet_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> RetinaNet: ...
def retinanet_resnet50_fpn_v2(*, weights: Optional[RetinaNet_ResNet50_FPN_V2_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> RetinaNet: ...
