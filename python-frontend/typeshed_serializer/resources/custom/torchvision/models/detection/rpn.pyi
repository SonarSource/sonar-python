import torch
from .anchor_utils import AnchorGenerator as AnchorGenerator
from .image_list import ImageList as ImageList
from torch import Tensor as Tensor, nn
from torchvision.ops import Conv2dNormActivation as Conv2dNormActivation
from typing import Any, Dict, List, Optional, Tuple

class RPNHead(nn.Module):
    conv: Any
    cls_logits: Any
    bbox_pred: Any
    def __init__(self, in_channels: int, num_anchors: int, conv_depth: int = ...) -> None: ...
    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]: ...

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor: ...
def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]: ...

class RegionProposalNetwork(torch.nn.Module):
    __annotations__: Any
    anchor_generator: Any
    head: Any
    box_coder: Any
    box_similarity: Any
    proposal_matcher: Any
    fg_bg_sampler: Any
    nms_thresh: Any
    score_thresh: Any
    min_size: float
    def __init__(self, anchor_generator: AnchorGenerator, head: nn.Module, fg_iou_thresh: float, bg_iou_thresh: float, batch_size_per_image: int, positive_fraction: float, pre_nms_top_n: Dict[str, int], post_nms_top_n: Dict[str, int], nms_thresh: float, score_thresh: float = ...) -> None: ...
    def pre_nms_top_n(self) -> int: ...
    def post_nms_top_n(self) -> int: ...
    def assign_targets_to_anchors(self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]: ...
    def filter_proposals(self, proposals: Tensor, objectness: Tensor, image_shapes: List[Tuple[int, int]], num_anchors_per_level: List[int]) -> Tuple[List[Tensor], List[Tensor]]: ...
    def compute_loss(self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]) -> Tuple[Tensor, Tensor]: ...
    def forward(self, images: ImageList, features: Dict[str, Tensor], targets: Optional[List[Dict[str, Tensor]]] = ...) -> Tuple[List[Tensor], Dict[str, Tensor]]: ...
