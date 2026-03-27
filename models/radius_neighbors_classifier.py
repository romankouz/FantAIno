import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from models.fantaino_base import FantAInoFitter

class RadiusNeighborsClassificationModel(FantAInoFitter):
    """Radius Neighbors Classifier model for FantAIno."""
    
    def __init__(
        self,
        radius: int = 5,
        param_grid: dict = None,
        scoring_method: str = "accuracy",
        model_run_name: str = "master",
        n_jobs: int = 4,
    ):
        super().__init__()

        self.param_grid = param_grid
        self.scoring_method = scoring_method
        self.n_jobs = n_jobs

        self.base_model = RadiusNeighborsClassifier(radius=radius, n_jobs=self.n_jobs)
        self.model = self.base_model
        self.model_name = "Radius Neighbors Classifier"
        self.model_run_name = model_run_name

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        
        if self.param_grid:
            self.model = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                scoring=self.scoring_method,
                n_jobs=self.n_jobs
            )

        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        try:
            check_is_fitted(self.model)
        # exc notation extends existing exception instead of overwriting it
        except NotFittedError as exc:
            raise NotFittedError("Model must be fitted before predicting.") from exc
        return self.model.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, loss_fn: accuracy_score) -> float:
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            raise NotFittedError("Model must be fitted before prediction and evaluation.") from exc
        return loss_fn(self.predict(X_test), y_test)


