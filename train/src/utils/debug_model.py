import logging
from torch import nn


def debug_model_grad(model: nn.Module, logger: logging.Logger):
    for name, param in model.named_parameters():
        logger.debug(f"{name} {param.grad}")
