import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score

from FantAIno.models.fantaino_base import FantAInoFitter

class NaiveMajorityModel(FantAInoFitter):
    """Naive majority predictions for FantAIno."""

    def __init__(self, model_run_name: str = "master"):
        super().__init__()
        self.model_name = "Naive Majority Guesser"
        self.model_run_name = model_run_name

    def train_model(self, X_train, y_train):
        # model is just the most common class
        self.model = stats.mode(y_train).mode.item()

    def predict_from_model(self, X_test):
        if self.model:
            return np.array([self.model] * len(X_test))
        else:
            raise ValueError("self.model is not fitted. Call train_model() first.")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, loss_fn: accuracy_score) -> float:
        return loss_fn(self.predict_from_model(X_test), y_test)