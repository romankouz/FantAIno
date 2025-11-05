from abc import ABC, abstractmethod

class FantAInoFitter(ABC):

    @property
    @abstractmethod
    def estaimtor(self):
        """The underlying model estimator."""

    @abstractmethod
    def train(input_data, response_data):
        """Training method"""

    @abstractmethod
    def predict(self, input_data):
        """Prediction method"""

    @abstractmethod
    def evaluate(self, input_data, response_data, loss_fn):
        """Evaluation method"""

    @abstractmethod
    def extract_features(
        self,
        dataset,
        feature_set,
        omit_mode=True,
    ):

    @abstractmethod
    def preprocess(self):
        """Preprcoessing steps that must occur for this model"""