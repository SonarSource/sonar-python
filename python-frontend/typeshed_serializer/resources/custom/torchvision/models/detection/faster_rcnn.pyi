from .._api import WeightsEnum
from ..mobilenetv3 import MobileNet_V3_Large_Weights
from ..resnet import ResNet50_Weights
from .generalized_rcnn import GeneralizedRCNN
from torch import nn
from typing import Any, Callable, List, Optional, Tuple

class FasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes: Any | None = ..., min_size: int = ..., max_size: int = ..., image_mean: Any | None = ..., image_std: Any | None = ..., rpn_anchor_generator: Any | None = ..., rpn_head: Any | None = ..., rpn_pre_nms_top_n_train: int = ..., rpn_pre_nms_top_n_test: int = ..., rpn_post_nms_top_n_train: int = ..., rpn_post_nms_top_n_test: int = ..., rpn_nms_thresh: float = ..., rpn_fg_iou_thresh: float = ..., rpn_bg_iou_thresh: float = ..., rpn_batch_size_per_image: int = ..., rpn_positive_fraction: float = ..., rpn_score_thresh: float = ..., box_roi_pool: Any | None = ..., box_head: Any | None = ..., box_predictor: Any | None = ..., box_score_thresh: float = ..., box_nms_thresh: float = ..., box_detections_per_img: int = ..., box_fg_iou_thresh: float = ..., box_bg_iou_thresh: float = ..., box_batch_size_per_image: int = ..., box_positive_fraction: float = ..., bbox_reg_weights: Any | None = ..., **kwargs) -> None: ...

class TwoMLPHead(nn.Module):
    fc6: Any
    fc7: Any
    def __init__(self, in_channels, representation_size) -> None: ...
    def forward(self, x): ...

class FastRCNNConvFCHead(nn.Sequential):
    def __init__(self, input_size: Tuple[int, int, int], conv_layers: List[int], fc_layers: List[int], norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...

class FastRCNNPredictor(nn.Module):
    cls_score: Any
    bbox_pred: Any
    def __init__(self, in_channels, num_classes) -> None: ...
    def forward(self, x): ...

class FasterRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

class FasterRCNN_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

class FasterRCNN_MobileNet_V3_Large_FPN_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

class FasterRCNN_MobileNet_V3_Large_320_FPN_Weights(WeightsEnum):
    COCO_V1: Any
    DEFAULT: Any

def fasterrcnn_resnet50_fpn(*, weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> FasterRCNN: ...
def fasterrcnn_resnet50_fpn_v2(*, weights: Optional[FasterRCNN_ResNet50_FPN_V2_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[ResNet50_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> FasterRCNN: ...
def fasterrcnn_mobilenet_v3_large_320_fpn(*, weights: Optional[FasterRCNN_MobileNet_V3_Large_320_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> FasterRCNN: ...
def fasterrcnn_mobilenet_v3_large_fpn(*, weights: Optional[FasterRCNN_MobileNet_V3_Large_FPN_Weights] = ..., progress: bool = ..., num_classes: Optional[int] = ..., weights_backbone: Optional[MobileNet_V3_Large_Weights] = ..., trainable_backbone_layers: Optional[int] = ..., **kwargs: Any) -> FasterRCNN: ...
