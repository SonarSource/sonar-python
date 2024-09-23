import torch.nn as nn
import torch.fx
from .._api import WeightsEnum
from typing import Any, Callable, List, Optional, Sequence, Tuple

class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]
    def __init__(self, num_heads, input_channels, output_channels, kernel_q, kernel_kv, stride_q, stride_kv) -> None: ...

class Pool(nn.Module):
    pool: Any
    norm_act: Any
    norm_before_pool: Any
    def __init__(self, pool: nn.Module, norm: Optional[nn.Module], activation: Optional[nn.Module] = ..., norm_before_pool: bool = ...) -> None: ...
    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]: ...

class MultiscaleAttention(nn.Module):
    embed_dim: Any
    output_dim: Any
    num_heads: Any
    head_dim: Any
    scaler: Any
    residual_pool: Any
    residual_with_cls_embed: Any
    qkv: Any
    project: Any
    pool_q: Any
    pool_k: Any
    pool_v: Any
    rel_pos_h: Any
    rel_pos_w: Any
    rel_pos_t: Any
    def __init__(self, input_size: List[int], embed_dim: int, output_dim: int, num_heads: int, kernel_q: List[int], kernel_kv: List[int], stride_q: List[int], stride_kv: List[int], residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, dropout: float = ..., norm_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]: ...

class MultiscaleBlock(nn.Module):
    proj_after_attn: Any
    pool_skip: Any
    norm1: Any
    norm2: Any
    needs_transposal: Any
    attn: Any
    mlp: Any
    stochastic_depth: Any
    project: Any
    def __init__(self, input_size: List[int], cnf: MSBlockConfig, residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, proj_after_attn: bool, dropout: float = ..., stochastic_depth_prob: float = ..., norm_layer: Callable[..., nn.Module] = ...) -> None: ...
    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]: ...

class PositionalEncoding(nn.Module):
    spatial_size: Any
    temporal_size: Any
    class_token: Any
    spatial_pos: Any
    temporal_pos: Any
    class_pos: Any
    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int, rel_pos_embed: bool) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MViT(nn.Module):
    conv_proj: Any
    pos_encoding: Any
    blocks: Any
    norm: Any
    head: Any
    def __init__(self, spatial_size: Tuple[int, int], temporal_size: int, block_setting: Sequence[MSBlockConfig], residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, proj_after_attn: bool, dropout: float = ..., attention_dropout: float = ..., stochastic_depth_prob: float = ..., num_classes: int = ..., block: Optional[Callable[..., nn.Module]] = ..., norm_layer: Optional[Callable[..., nn.Module]] = ..., patch_embed_kernel: Tuple[int, int, int] = ..., patch_embed_stride: Tuple[int, int, int] = ..., patch_embed_padding: Tuple[int, int, int] = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MViT_V1_B_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

class MViT_V2_S_Weights(WeightsEnum):
    KINETICS400_V1: Any
    DEFAULT: Any

def mvit_v1_b(*, weights: Optional[MViT_V1_B_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MViT: ...
def mvit_v2_s(*, weights: Optional[MViT_V2_S_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MViT: ...
