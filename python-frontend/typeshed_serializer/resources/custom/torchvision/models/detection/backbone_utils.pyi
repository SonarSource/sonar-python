from .. import mobilenet as mobilenet, resnet as resnet
from .._api import WeightsEnum as WeightsEnum
from .._utils import IntermediateLayerGetter as IntermediateLayerGetter, handle_legacy_interface as handle_legacy_interface
from torch import Tensor as Tensor, nn
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock as ExtraFPNBlock, FeaturePyramidNetwork as FeaturePyramidNetwork, LastLevelMaxPool as LastLevelMaxPool
from typing import Any, Callable, Dict, List, Optional

class BackboneWithFPN(nn.Module):
    body: Any
    fpn: Any
    out_channels: Any
    def __init__(self, backbone: nn.Module, return_layers: Dict[str, str], in_channels_list: List[int], out_channels: int, extra_blocks: Optional[ExtraFPNBlock] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Dict[str, Tensor]: ...

def resnet_fpn_backbone(*, backbone_name: str, weights: Optional[WeightsEnum], norm_layer: Callable[..., nn.Module] = ..., trainable_layers: int = ..., returned_layers: Optional[List[int]] = ..., extra_blocks: Optional[ExtraFPNBlock] = ...) -> BackboneWithFPN: ...
def mobilenet_backbone(*, backbone_name: str, weights: Optional[WeightsEnum], fpn: bool, norm_layer: Callable[..., nn.Module] = ..., trainable_layers: int = ..., returned_layers: Optional[List[int]] = ..., extra_blocks: Optional[ExtraFPNBlock] = ...) -> nn.Module: ...
