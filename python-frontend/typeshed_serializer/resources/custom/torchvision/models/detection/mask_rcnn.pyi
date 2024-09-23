from .._api import WeightsEnum
from ..resnet import ResNet50_Weights
from .faster_rcnn import FasterRCNN
from torch import nn
from typing import Any, Callable, Optional

class MaskRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes: Any | None = ..., min_size: int = ..., max_size: int = ..., image_mean: Any | None = ..., image_std: Any | None = ..., rpn_anchor_generator: Any | None = ..., rpn_head: Any | None = ..., rpn_pre_nms_top_n_train: int = ..., rpn_pre_nms_top_n_test: int = ..., rpn_post_nms_top_n_train: int = ..., rpn_post_nms_top_n_test: int = ..., rpn_nms_thresh: float = ..., rpn_fg_iou_thresh: float = ..., rpn_bg_iou_thresh: float = ..., rpn_batch_size_per_image: int = ..., rpn_positive_fraction: float = ..., rpn_score_thresh: float = ..., box_roi_pool: Any | None = ..., box_head: Any | None = ..., box_predictor: Any | None = ..., box_score_thresh: float = ..., box_nms_thresh: float = ..., box_detections_per_img: int = ..., box_fg_iou_thresh: float = ..., box_bg_iou_thresh: float = ..., box_batch_size_per_image: int = ..., box_positive_fraction: float = ..., bbox_reg_weights: Any | None = ..., mask_roi_pool: Any | None = ..., mask_head: Any | None = ..., mask_predictor: Any | None = ..., **kwargs) -> None: ...

class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation, norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...

class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes) -> None: ...

class MaskRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

class MaskRCNN_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

def maskrcnn_resnet50_fpn(*, weights: Optional[MaskRCNN_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> MaskRCNN: ...
def maskrcnn_resnet50_fpn_v2(*, weights: Optional[MaskRCNN_ResNet50_FPN_V2_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> MaskRCNN: ...
