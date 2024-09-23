from torch import Tensor as Tensor, nn
from typing import Any, Dict, List, Optional, Tuple, Union

class GeneralizedRCNN(nn.Module):
    transform: Any
    backbone: Any
    rpn: Any
    roi_heads: Any
    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None: ...
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...
    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = ...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]: ...
