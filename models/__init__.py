from .d_projector import DProjector
from .backbone import VisualEncoder
from .token_mlp import TokenConnector
from .mask import MaskGenerator
from .bev import BEVTransformer, BEVLinearAttention
from .encoder import DualInputVisualEncoder
from .feature_diff import FeatureDiffModule

def build_textencoder(cfg):
    if cfg.model.model_name == "bert-base-uncased":
        from .bert import TextEncoder
    elif cfg.model.model_name == "microsoft/phi-1_5":
        from .llm import TextEncoder
    else:
        raise NotImplementedError
    return TextEncoder


from .segchange import build_model
