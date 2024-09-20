from .._api import WeightsEnum
from ..resnet import ResNet50_Weights
from .faster_rcnn import FasterRCNN
from torch import nn
from typing import Any, Optional

class KeypointRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes: Any | None = ..., min_size: Any | None = ..., max_size: int = ..., image_mean: Any | None = ..., image_std: Any | None = ..., rpn_anchor_generator: Any | None = ..., rpn_head: Any | None = ..., rpn_pre_nms_top_n_train: int = ..., rpn_pre_nms_top_n_test: int = ..., rpn_post_nms_top_n_train: int = ..., rpn_post_nms_top_n_test: int = ..., rpn_nms_thresh: float = ..., rpn_fg_iou_thresh: float = ..., rpn_bg_iou_thresh: float = ..., rpn_batch_size_per_image: int = ..., rpn_positive_fraction: float = ..., rpn_score_thresh: float = ..., box_roi_pool: Any | None = ..., box_head: Any | None = ..., box_predictor: Any | None = ..., box_score_thresh: float = ..., box_nms_thresh: float = ..., box_detections_per_img: int = ..., box_fg_iou_thresh: float = ..., box_bg_iou_thresh: float = ..., box_batch_size_per_image: int = ..., box_positive_fraction: float = ..., bbox_reg_weights: Any | None = ..., keypoint_roi_pool: Any | None = ..., keypoint_head: Any | None = ..., keypoint_predictor: Any | None = ..., num_keypoints: Any | None = ..., **kwargs) -> None: ...

class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers) -> None: ...

class KeypointRCNNPredictor(nn.Module):
    kps_score_lowres: Any
    up_scale: int
    out_channels: Any
    def __init__(self, in_channels, num_keypoints) -> None: ...
    def forward(self, x): ...

class KeypointRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_LEGACY: Any
    COCO_V1: Any
    DEFAULT: Any

def keypointrcnn_resnet50_fpn(*, weights: Optional[KeypointRCNN_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., num_keypoints: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> KeypointRCNN: ...
