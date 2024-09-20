import torch.nn as nn
import torch
from ..ops.misc import MLP
from ._api import WeightsEnum
from typing import Any, Callable, List, NamedTuple, Optional

class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module]
    activation_layer: Callable[..., nn.Module]

class MLPBlock(MLP):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float) -> None: ...

class EncoderBlock(nn.Module):
    num_heads: Any
    ln_1: Any
    self_attention: Any
    dropout: Any
    ln_2: Any
    mlp: Any
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float, norm_layer: Callable[..., torch.nn.Module] = ...) -> None: ...
    def forward(self, input: torch.Tensor): ...

class Encoder(nn.Module):
    pos_embedding: Any
    dropout: Any
    layers: Any
    ln: Any
    def __init__(self, seq_length: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float, norm_layer: Callable[..., torch.nn.Module] = ...) -> None: ...
    def forward(self, input: torch.Tensor): ...

class VisionTransformer(nn.Module):
    image_size: Any
    patch_size: Any
    hidden_dim: Any
    mlp_dim: Any
    attention_dropout: Any
    dropout: Any
    num_classes: Any
    representation_size: Any
    norm_layer: Any
    conv_proj: Any
    class_token: Any
    encoder: Any
    seq_length: Any
    heads: Any
    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float = ..., attention_dropout: float = ..., num_classes: int = ..., representation_size: Optional[int] = ..., norm_layer: Callable[..., torch.nn.Module] = ..., conv_stem_configs: Optional[List[ConvStemConfig]] = ...) -> None: ...
    def forward(self, x: torch.Tensor): ...

class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_SWAG_E2E_V1: Any
    IMAGENET1K_SWAG_LINEAR_V1: Any
    DEFAULT: Any

class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    IMAGENET1K_SWAG_E2E_V1: Any
    IMAGENET1K_SWAG_LINEAR_V1: Any
    DEFAULT: Any

class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1: Any
    DEFAULT: Any

class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1: Any
    IMAGENET1K_SWAG_LINEAR_V1: Any
    DEFAULT: Any

def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer: ...
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer: ...
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer: ...
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer: ...
def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer: ...
