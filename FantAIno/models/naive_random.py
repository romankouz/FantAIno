import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from FantAIno.models.fantaino_base import FantAInoFitter

class NaiveRandomModel(FantAInoFitter):
    """Naive random predictions for FantAIno."""

    def __init__(self, model_run_name: str = "master"):
        super().__init__()
        self.model_name = "Naive Random Guesser"
        self.model_run_name = model_run_name
        self.model = "100 horses. Confused right? Well it's random :). No model for random guessing."

    def train_model(self, X_train, y_train):
        print("Naive random guesser doesn't need to be trained! We just randomly guess :D")

    def predict_from_model(self, X_test: pd.DataFrame) -> pd.Series:
        return np.random.choice(np.arange(-1, 11), len(X_test))

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, loss_fn: accuracy_score) -> float:
        return loss_fn(self.predict_from_model(X_test), y_test)