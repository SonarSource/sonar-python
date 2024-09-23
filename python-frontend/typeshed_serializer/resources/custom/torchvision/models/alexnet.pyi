import torch.nn as nn
import torch
from ._api import WeightsEnum
from typing import Any, Optional

class AlexNet(nn.Module):
    features: Any
    avgpool: Any
    classifier: Any
    def __init__(self, num_classes: int = ..., dropout: float = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AlexNet_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def alexnet(*, weights: Optional[AlexNet_Weights] = ..., progress: bool = ..., **kwargs: Any) -> AlexNet: ...
