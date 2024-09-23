from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, NamedTuple, Optional

class InceptionOutputs(NamedTuple):
    logits: Any
    aux_logits: Any

class Inception3(nn.Module):
    aux_logits: Any
    transform_input: Any
    Conv2d_1a_3x3: Any
    Conv2d_2a_3x3: Any
    Conv2d_2b_3x3: Any
    maxpool1: Any
    Conv2d_3b_1x1: Any
    Conv2d_4a_3x3: Any
    maxpool2: Any
    Mixed_5b: Any
    Mixed_5c: Any
    Mixed_5d: Any
    Mixed_6a: Any
    Mixed_6b: Any
    Mixed_6c: Any
    Mixed_6d: Any
    Mixed_6e: Any
    AuxLogits: Any
    Mixed_7a: Any
    Mixed_7b: Any
    Mixed_7c: Any
    avgpool: Any
    dropout: Any
    fc: Any
    def __init__(self, num_classes: int = ..., aux_logits: bool = ..., transform_input: bool = ..., inception_blocks: Optional[List[Callable[..., nn.Module]]] = ..., init_weights: Optional[bool] = ..., dropout: float = ...) -> None: ...
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs: ...
    def forward(self, x: Tensor) -> InceptionOutputs: ...

class InceptionA(nn.Module):
    branch1x1: Any
    branch5x5_1: Any
    branch5x5_2: Any
    branch3x3dbl_1: Any
    branch3x3dbl_2: Any
    branch3x3dbl_3: Any
    branch_pool: Any
    def __init__(self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class InceptionB(nn.Module):
    branch3x3: Any
    branch3x3dbl_1: Any
    branch3x3dbl_2: Any
    branch3x3dbl_3: Any
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class InceptionC(nn.Module):
    branch1x1: Any
    branch7x7_1: Any
    branch7x7_2: Any
    branch7x7_3: Any
    branch7x7dbl_1: Any
    branch7x7dbl_2: Any
    branch7x7dbl_3: Any
    branch7x7dbl_4: Any
    branch7x7dbl_5: Any
    branch_pool: Any
    def __init__(self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class InceptionD(nn.Module):
    branch3x3_1: Any
    branch3x3_2: Any
    branch7x7x3_1: Any
    branch7x7x3_2: Any
    branch7x7x3_3: Any
    branch7x7x3_4: Any
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class InceptionE(nn.Module):
    branch1x1: Any
    branch3x3_1: Any
    branch3x3_2a: Any
    branch3x3_2b: Any
    branch3x3dbl_1: Any
    branch3x3dbl_2: Any
    branch3x3dbl_3a: Any
    branch3x3dbl_3b: Any
    branch_pool: Any
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class InceptionAux(nn.Module):
    conv0: Any
    conv1: Any
    fc: Any
    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class BasicConv2d(nn.Module):
    conv: Any
    bn: Any
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Inception_V3_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def inception_v3(*, weights: Optional[Inception_V3_Weights] = ..., progress: bool = ..., **kwargs: Any) -> Inception3: ...

# Names in __all__ with no definition:
#   InceptionOutputs
#   _InceptionOutputs
