import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from catboost import CatBoostRegressor, CatBoostClassifier

from FantAIno.models.fantaino_base import FantAInoFitter

class CatBoostModel(FantAInoFitter):
    """CatBoost model for FantAIno."""

    def __init__(
        self,
        mode: str = "regression",
        iterations: int = 100,
        depth: int = 3,
        loss_fn: str = "RMSE",
        learning_rate: float = 0.01,
        param_grid: dict = None,
        scoring_method: str = "neg_mean_squared_error",
        model_run_name: str = "master",
        n_jobs: int = 4,
    ):
        super().__init__()

        self.mode = mode
        self.param_grid = param_grid
        self.n_jobs = n_jobs

        self.loss_fn = loss_fn
        self.scoring_method = scoring_method

        if self.mode == "regression":
            self.base_model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                loss_function=self.loss_fn,
                learning_rate=learning_rate
            )
            self.model_name = "CatBoost Regressor"
        elif self.mode == "classification":
            self.base_model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                loss_function=self.loss_fn,
                learning_rate=learning_rate
            )
            self.model_name = "CatBoost Classifier"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.model = self.base_model
        self.model_run_name = model_run_name

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.param_grid:
            self.model = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                scoring=self.scoring_method,
                n_jobs=self.n_jobs
            )
        self.model.fit(X_train, y_train)

    def predict_from_model(self, X_test: pd.DataFrame) -> pd.Series:
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            raise NotFittedError("Model must be fitted before predicting.") from exc
        return self.model.predict(X_test)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, loss_fn: callable = None) -> float:
        if loss_fn is None:
            loss_fn = mean_squared_error if self.mode == "regression" else accuracy_score
        try:
            check_is_fitted(self.model)
        except NotFittedError as exc:
            raise NotFittedError("Model must be fitted before prediction and evaluation.") from exc
        preds = self.predict_from_model(X_test)
        return loss_fn(preds, y_test)
