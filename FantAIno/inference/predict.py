"""
The official prediction script for FantAIno.
"""
import datetime
import os

from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from FantAIno.constants import RESULTS_DIR
from FantAIno.utils.metrics import rounded_regression_accuracy

# predict using a trainedmodel
def predict(cfg: DictConfig) -> None:
    """Get predictions from a trained FantAIno model."""

    album_data_preprocessing_pipeline = instantiate(cfg.preprocessing.album_data_preprocessing_pipeline)
    lyrics_preprocessing_pipeline = instantiate(cfg.preprocessing.lyrics_preprocessing_pipeline)
    FantAIno_df_obj = instantiate(
        cfg.dataset,
        album_data_preprocessing_pipeline=album_data_preprocessing_pipeline,
        lyrics_preprocessing_pipeline=lyrics_preprocessing_pipeline
    )

    X_test, y_test = FantAIno_df_obj.FantAIno_df_X, FantAIno_df_obj.FantAIno_df_y

    model = instantiate(cfg.model, _convert_="partial")
    model.load_estimator(model.model_run_name)
    y_pred = model.predict_from_model(X_test)

    match cfg.prediction.mode:

        case "predict":
            return y_pred

        case "evaluate":

            if cfg.prediction.record_result:
                
                # change the score to use what the model uses
                if any(term in model.model_name for term in ["Ordinal Logistic", "Guesser", "Classifier"]):
                    metric = "accuracy"
                    test_loss = model.evaluate_model(X_test, y_test)
                else:
                    metric = "MSE"
                    test_loss = model.evaluate_model(X_test, y_test)

                result = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model.model_name,
                    "run_name": model.model_run_name,
                    "dataset": str(cfg.dataset._target_).rsplit('.', maxsplit=1)[-1],
                    "metric": metric,
                    "score": test_loss
                }

                if metric != "accuracy":
                    rounded_accuracy_entry = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model.model_name,
                        "run_name": model.model_run_name,
                        "dataset": str(cfg.dataset._target_).rsplit('.', maxsplit=1)[-1],
                        "metric": "accuracy",
                        "score": rounded_regression_accuracy(y_test, y_pred)
                    }
                    result = [result, rounded_accuracy_entry]

                results_path = os.path.join(RESULTS_DIR, "all_results.csv")
                result_df = pd.DataFrame(list(result))

                if os.path.exists(results_path):
                    existing_df = pd.read_csv(results_path)
                    merged_df = pd.concat([existing_df, result_df], ignore_index=True)
                    merged_df = merged_df.drop_duplicates(subset=['model', 'run_name', 'dataset', 'metric'], keep='last')
                else:
                    merged_df = result_df

                merged_df.to_csv(results_path, index=False)

            print(f"{model.model_name} Test {metric}: {test_loss}")
            predictions_path = os.path.join(RESULTS_DIR, f"{model.model_name}_{model.model_run_name}.csv")
            predictions_df = pd.concat([FantAIno_df_obj.FantAIno_index.to_frame().reset_index(drop=True), pd.Series(y_pred.ravel()), pd.Series(y_test.ravel())], axis=1)
            predictions_df.columns = ["album_name", "artist_name", "prediction", "actual"]
            predictions_df.to_csv(predictions_path)

            return y_pred, test_loss

        case _:
            raise ValueError(f"Invalid mode: {cfg.prediction.mode}")
