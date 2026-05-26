from abc import ABC, abstractmethod
import joblib
import os

import FantAIno

class FantAInoFitter(ABC):
    """Abstract base class for all FantAIno models."""

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
    def train_model(self, X_train, y_train):
        """Training method (named to avoid clashing with nn.Module.train and similar)."""

    @abstractmethod
    def predict_from_model(self, X_test):
        """Prediction method (named to avoid clashing with estimator .predict APIs)."""

    @abstractmethod
    def evaluate_model(self, X_test, y_test, loss_fn):
        """Evaluation method (named to avoid clashing with other evaluate helpers)."""