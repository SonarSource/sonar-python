import torch
from ._api import WeightsEnum
from torch import Tensor, nn
from typing import Any, Callable, List, Optional

class PatchMerging(nn.Module):
    dim: Any
    reduction: Any
    norm: Any
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor): ...

class PatchMergingV2(nn.Module):
    dim: Any
    reduction: Any
    norm: Any
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor): ...

class ShiftedWindowAttention(nn.Module):
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
    def get_relative_position_bias(self) -> torch.Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...

class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    logit_scale: Any
    cpb_mlp: Any
    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool = ..., proj_bias: bool = ..., attention_dropout: float = ..., dropout: float = ...) -> None: ...
    def define_relative_position_bias_table(self) -> None: ...
    def get_relative_position_bias(self) -> torch.Tensor: ...
    def forward(self, x: Tensor): ...

class SwinTransformerBlock(nn.Module):
    norm1: Any
    attn: Any
    stochastic_depth: Any
    norm2: Any
    mlp: Any
    def __init__(self, dim: int, num_heads: int, window_size: List[int], shift_size: List[int], mlp_ratio: float = ..., dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., norm_layer: Callable[..., nn.Module] = ..., attn_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor): ...

class SwinTransformerBlockV2(SwinTransformerBlock):
    def __init__(self, dim: int, num_heads: int, window_size: List[int], shift_size: List[int], mlp_ratio: float = ..., dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., norm_layer: Callable[..., nn.Module] = ..., attn_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: Tensor): ...

class SwinTransformer(nn.Module):
    num_classes: Any
    features: Any
    norm: Any
    permute: Any
    avgpool: Any
    flatten: Any
    head: Any
    def __init__(self, patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], mlp_ratio: float = ..., dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., num_classes: int = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., block: Optional[Callable[..., nn.Module]] = ..., downsample_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x): ...

class Swin_T_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class Swin_S_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class Swin_B_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class Swin_V2_T_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class Swin_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class Swin_V2_B_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def swin_t(*, weights: Optional[Swin_T_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer: ...
def swin_s(*, weights: Optional[Swin_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer: ...
def swin_b(*, weights: Optional[Swin_B_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer: ...
def swin_v2_t(*, weights: Optional[Swin_V2_T_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer: ...
def swin_v2_s(*, weights: Optional[Swin_V2_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer: ...
def swin_v2_b(*, weights: Optional[Swin_V2_B_Weights] = ..., progress: bool = ..., **kwargs: Any) -> SwinTransformer: ...
