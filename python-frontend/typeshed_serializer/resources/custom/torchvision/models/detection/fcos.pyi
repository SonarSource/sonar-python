from .._api import WeightsEnum
from ..resnet import ResNet50_Weights
from .anchor_utils import AnchorGenerator
from torch import Tensor, nn
from typing import Any, Callable, Dict, List, Optional, Tuple

class FCOSHead(nn.Module):
    __annotations__: Any
    box_coder: Any
    classification_head: Any
    regression_head: Any
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: Optional[int] = ...) -> None: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]: ...
    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]: ...

class FCOSClassificationHead(nn.Module):
    num_classes: Any
    num_anchors: Any
    conv: Any
    cls_logits: Any
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int = ..., prior_probability: float = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: List[Tensor]) -> Tensor: ...

class FCOSRegressionHead(nn.Module):
    conv: Any
    bbox_reg: Any
    bbox_ctrness: Any
    def __init__(self, in_channels: int, num_anchors: int, num_convs: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]: ...

class FCOS(nn.Module):
    __annotations__: Any
    backbone: Any
    anchor_generator: Any
    head: Any
    box_coder: Any
    transform: Any
    center_sampling_radius: Any
    score_thresh: Any
    nms_thresh: Any
    detections_per_img: Any
    topk_candidates: Any
    def __init__(self, backbone: nn.Module, num_classes: int, min_size: int = ..., max_size: int = ..., image_mean: Optional[List[float]] = ..., image_std: Optional[List[float]] = ..., anchor_generator: Optional[AnchorGenerator] = ..., head: Optional[nn.Module] = ..., center_sampling_radius: float = ..., score_thresh: float = ..., nms_thresh: float = ..., detections_per_img: int = ..., topk_candidates: int = ..., **kwargs) -> None: ...
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...
    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], num_anchors_per_level: List[int]) -> Dict[str, Tensor]: ...
    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]: ...
    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = ...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...

class FCOS_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

def fcos_resnet50_fpn(*, weights: Optional[FCOS_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> FCOS: ...
