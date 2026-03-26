import joblib
import os
import pandas as pd

from abc import ABC, abstractmethod

import FantAIno

class FantAInoFitter(ABC):

    def __init__(self):
        self.model = None
        self.model_name = "fantaino_model"
        self.root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))


    def load_estimator(self, model_run_name: str):
        """Retrieve the current estimator."""
        self.model = joblib.load(os.path.join(self.root_dir, "results", self.model_name, f"{model_run_name}.joblib"))


    def save_estimator(self, model_run_name: str):
        """Save the current estimator to a file."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_path = os.path.join(self.root_dir, "results", self.model_name)
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_path, f"{model_run_name}.joblib"))


    @abstractmethod
    def train(self, X_train, y_train):
        """Training method."""

    @abstractmethod
    def predict(self, X_test):
        """Prediction method."""

    @abstractmethod
    def evaluate(self, X_test, y_test, loss_fn):
        """Evaluation method."""