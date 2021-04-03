__version__ = "0.7.0"
from .model import EfficientNet
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']


def efficientnet_b0(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name('efficientnet-b0', **kwargs)


def efficientnet_b1(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name('efficientnet-b1', **kwargs)


def efficientnet_b2(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name('efficientnet-b2', **kwargs)


def efficientnet_b3(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name('efficientnet-b3', **kwargs)

