import torch
from .._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, Optional

class ShiftedWindowAttention3d(nn.Module):
    window_size: Any
    shift_size: Any
    num_heads: Any
    attention_dropout: Any
    dropout: Any
    qkv: Any
    proj: Any
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool = ..., proj_bias: bool = ..., attention_dropout: float = ..., dropout: float = ...) -> None: ...
    relative_position_bias_table: Any
    def define_relative_position_bias_table(self) -> None: ...
    def define_relative_position_index(self) -> None: ...
    def get_relative_position_bias(self, window_size: List[int]) -> torch.Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...

class PatchEmbed3d(nn.Module):
    tuple_patch_size: Any
    proj: Any
    norm: Any
    def __init__(self, patch_size: List[int], in_channels: int = ..., embed_dim: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class SwinTransformer3d(nn.Module):
    num_classes: Any
    patch_embed: Any
    pos_drop: Any
    features: Any
    num_features: Any
    norm: Any
    avgpool: Any
    head: Any
    def __init__(self, patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], mlp_ratio: float = ..., dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., num_classes: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., block: Optional[Callable[..., nn.Module]] = ..., downsample_layer: Callable[..., nn.Module] = ..., patch_embed: Optional[Callable[..., nn.Module]] = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class Swin3D_T_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

class Swin3D_S_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

class Swin3D_B_Weights(WeightsEnum):
    KINETICS400_V1: Any
    KINETICS400_IMAGENET22K_V1: Any
    DEFAULT: Any

def swin3d_t(*, weights: Optional[Swin3D_T_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer3d: ...
def swin3d_s(*, weights: Optional[Swin3D_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer3d: ...
def swin3d_b(*, weights: Optional[Swin3D_B_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer3d: ...
