from torch import Tensor as Tensor, nn
from typing import Any, Dict, Optional

class _SimpleSegmentationModel(nn.Module):
    __constants__: Any
    backbone: Any
    classifier: Any
    aux_classifier: Any
    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor) -> Dict[str, Tensor]: ...
