import torch
from torch import Tensor as Tensor
from typing import Any, List, Tuple
from SonarPythonAnalyzerFakeStub import CustomStubBase

class ImageList(CustomStubBase):
    tensors: Any
    image_sizes: Any
    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None: ...
    def to(self, device: torch.device) -> ImageList: ...
