import tensorflow as tf
import torch

from FantAIno.models.fantaino_base import FantAInoFitter

class NeuralNetworkModel(FantAInoFitter):
    """Dense Neural Network model for FantAIno."""

    def __init__(
        self,
    ):
        super().__init__()