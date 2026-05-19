from math import e
import numpy as np
from scipy import stats

from FantAIno.models.fantaino_base import FantAInoFitter

def NaiveMajorityModel(FantAInoFitter):
    """Naive majority predictions for FantAIno."""

    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train):
        # model is just the most common class
        self.model = stats.mode(y_train).mode[0]

    def predict(self, X_test):
        if self.model is not None:
            return np.array([self.model] * len(X_test))
        else:
            raise ValueError(f"self.model is not fitted. Call train() first.")