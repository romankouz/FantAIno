import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from models.fantaino_base import FantAInoFitter

class KNNClassifier(FantAInoFitter):

    def __init__(
        self,
        n_neighbors: int = 5,
        param_grid: dict = None,
        scoring_method: str = "accuracy",
        model_run_name: str = "master"
    ):
        super().__init__()

        self.param_grid = param_grid
        self.scoring_method = scoring_method

        self.base_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model = self.base_model
        self.model_name = "KNN Classifier"
        self.model_run_name = model_run_name

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        
        if self.param_grid:
            self.model = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                scoring=self.scoring_method
            )

        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise NotFittedError("Model must be fitted before predicting.")
        return self.model.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, loss_fn: accuracy_score) -> float:
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise NotFittedError("Model must be fitted before prediction and evaluation.")
        return loss_fn(self.predict(X_test), y_test)


