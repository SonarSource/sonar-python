import torch
from torch import Tensor, nn
from torchvision.models._api import WeightsEnum
from typing import Any, Callable, List, Optional, Tuple

class MBConv(nn.Module):
    proj: Any
    stochastic_depth: Any
    layers: Any
    def __init__(self, in_channels: int, out_channels: int, expansion_ratio: float, squeeze_ratio: float, stride: int, activation_layer: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], p_stochastic_dropout: float = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class RelativePositionalMultiHeadAttention(nn.Module):
    n_heads: Any
    head_dim: Any
    size: Any
    max_seq_len: Any
    to_qkv: Any
    scale_factor: Any
    merge: Any
    relative_position_bias_table: Any
    def __init__(self, feat_dim: int, head_dim: int, max_seq_len: int) -> None: ...
    def get_relative_positional_bias(self) -> torch.Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...

class SwapAxes(nn.Module):
    a: Any
    b: Any
    def __init__(self, a: int, b: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class WindowPartition(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, x: Tensor, p: int) -> Tensor: ...

class WindowDepartition(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, x: Tensor, p: int, h_partitions: int, w_partitions: int) -> Tensor: ...

class PartitionAttentionLayer(nn.Module):
    n_heads: Any
    head_dim: Any
    n_partitions: Any
    partition_type: Any
    grid_size: Any
    partition_op: Any
    departition_op: Any
    partition_swap: Any
    departition_swap: Any
    attn_layer: Any
    mlp_layer: Any
    stochastic_dropout: Any
    def __init__(self, in_channels: int, head_dim: int, partition_size: int, partition_type: str, grid_size: Tuple[int, int], mlp_ratio: int, activation_layer: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], attention_dropout: float, mlp_dropout: float, p_stochastic_dropout: float) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MaxVitLayer(nn.Module):
    layers: Any
    def __init__(self, in_channels: int, out_channels: int, squeeze_ratio: float, expansion_ratio: float, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], head_dim: int, mlp_ratio: int, mlp_dropout: float, attention_dropout: float, p_stochastic_dropout: float, partition_size: int, grid_size: Tuple[int, int]) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MaxVitBlock(nn.Module):
    layers: Any
    grid_size: Any
    def __init__(self, in_channels: int, out_channels: int, squeeze_ratio: float, expansion_ratio: float, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], head_dim: int, mlp_ratio: int, mlp_dropout: float, attention_dropout: float, partition_size: int, input_grid_size: Tuple[int, int], n_layers: int, p_stochastic: List[float]) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MaxVit(nn.Module):
    stem: Any
    partition_size: Any
    blocks: Any
    classifier: Any
    def __init__(self, input_size: Tuple[int, int], stem_channels: int, partition_size: int, block_channels: List[int], block_layers: List[int], head_dim: int, stochastic_depth_prob: float, norm_layer: Optional[Callable[..., nn.Module]] = ..., activation_layer: Callable[..., nn.Module] = ..., squeeze_ratio: float = ..., expansion_ratio: float = ..., mlp_ratio: int = ..., mlp_dropout: float = ..., attention_dropout: float = ..., num_classes: int = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MaxVit_T_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

def maxvit_t(*, weights: Optional[MaxVit_T_Weights] = ..., progress: bool = ..., **kwargs: Any) -> MaxVit: ...
