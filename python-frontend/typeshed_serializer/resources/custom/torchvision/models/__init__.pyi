from .alexnet import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .googlenet import *
from .inception import *
from .mnasnet import *
from .mobilenet import *
from .regnet import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .vgg import *
from .vision_transformer import *
from .swin_transformer import *
from .maxvit import *
from . import detection as detection, optical_flow as optical_flow, quantization as quantization, segmentation as segmentation, video as video
from ._api import Weights as Weights, WeightsEnum as WeightsEnum, get_model as get_model, get_model_builder as get_model_builder, get_model_weights as get_model_weights, get_weight as get_weight, list_models as list_models
