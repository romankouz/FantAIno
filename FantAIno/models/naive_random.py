import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from FantAIno.models.fantaino_base import FantAInoFitter

class NaiveRandomModel(FantAInoFitter):
    """Naive random predictions for FantAIno."""

    def __init__(self):
        super().__init__()

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return np.random.choice(np.arange(-1, 11), len(X_test))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, loss_fn: accuracy_score) -> float:
        return loss_fn(self.predict(X_test), y_test)