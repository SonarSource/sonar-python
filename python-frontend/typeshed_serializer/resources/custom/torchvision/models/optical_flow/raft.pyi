import torch.nn as nn
from .._api import WeightsEnum
from typing import Any, Optional

class ResidualBlock(nn.Module):
    convnormrelu1: Any
    convnormrelu2: Any
    downsample: Any
    relu: Any
    def __init__(self, in_channels, out_channels, *, norm_layer, stride: int = ..., always_project: bool = ...) -> None: ...
    def forward(self, x): ...

class BottleneckBlock(nn.Module):
    convnormrelu1: Any
    convnormrelu2: Any
    convnormrelu3: Any
    relu: Any
    downsample: Any
    def __init__(self, in_channels, out_channels, *, norm_layer, stride: int = ...) -> None: ...
    def forward(self, x): ...

class FeatureEncoder(nn.Module):
    convnormrelu: Any
    layer1: Any
    layer2: Any
    layer3: Any
    conv: Any
    output_dim: Any
    downsample_factor: Any
    def __init__(self, *, block=..., layers=..., strides=..., norm_layer=...): ...
    def forward(self, x): ...

class MotionEncoder(nn.Module):
    convcorr1: Any
    convcorr2: Any
    convflow1: Any
    convflow2: Any
    conv: Any
    out_channels: Any
    def __init__(self, *, in_channels_corr, corr_layers=..., flow_layers=..., out_channels: int = ...) -> None: ...
    def forward(self, flow, corr_features): ...

class ConvGRU(nn.Module):
    convz: Any
    convr: Any
    convq: Any
    def __init__(self, *, input_size, hidden_size, kernel_size, padding) -> None: ...
    def forward(self, h, x): ...

class RecurrentBlock(nn.Module):
    convgru1: Any
    convgru2: Any
    hidden_size: Any
    def __init__(self, *, input_size, hidden_size, kernel_size=..., padding=...) -> None: ...
    def forward(self, h, x): ...

class FlowHead(nn.Module):
    conv1: Any
    conv2: Any
    relu: Any
    def __init__(self, *, in_channels, hidden_size) -> None: ...
    def forward(self, x): ...

class UpdateBlock(nn.Module):
    motion_encoder: Any
    recurrent_block: Any
    flow_head: Any
    hidden_state_size: Any
    def __init__(self, *, motion_encoder, recurrent_block, flow_head) -> None: ...
    def forward(self, hidden_state, context, corr_features, flow): ...

class MaskPredictor(nn.Module):
    convrelu: Any
    conv: Any
    multiplier: Any
    def __init__(self, *, in_channels, hidden_size, multiplier: float = ...) -> None: ...
    def forward(self, x): ...

class CorrBlock(nn.Module):
    num_levels: Any
    radius: Any
    corr_pyramid: Any
    out_channels: Any
    def __init__(self, *, num_levels: int = ..., radius: int = ...) -> None: ...
    def build_pyramid(self, fmap1, fmap2) -> None: ...
    def index_pyramid(self, centroids_coords): ...

class RAFT(nn.Module):
    feature_encoder: Any
    context_encoder: Any
    corr_block: Any
    update_block: Any
    mask_predictor: Any
    def __init__(self, *, feature_encoder, context_encoder, corr_block, update_block, mask_predictor: Any | None = ...) -> None: ...
    def forward(self, image1, image2, num_flow_updates: int = ...): ...

class Raft_Large_Weights(WeightsEnum):
    C_T_V1: Any
    C_T_V2: Any
    C_T_SKHT_V1: Any
    C_T_SKHT_V2: Any
    C_T_SKHT_K_V1: Any
    C_T_SKHT_K_V2: Any
    DEFAULT: Any

class Raft_Small_Weights(WeightsEnum):
    C_T_V1: Any
    C_T_V2: Any
    DEFAULT: Any

def raft_large(*, weights: Optional[Raft_Large_Weights] = ..., progress: bool = ..., **kwargs) -> RAFT: ...
def raft_small(*, weights: Optional[Raft_Small_Weights] = ..., progress: bool = ..., **kwargs) -> RAFT: ...
