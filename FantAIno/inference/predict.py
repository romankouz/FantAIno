from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import FantAIno

# predict using a trainedmodel
def predict(cfg: DictConfig) -> None:
    """Get predictions from a trained FantAIno model."""

    FantAIno_df_obj = instantiate(cfg.dataset)

    X_test, y_test = FantAIno_df_obj.FantAIno_df_X, FantAIno_df_obj.FantAIno_df_y
    
    model = instantiate(cfg.model, _convert_="partial")
    model.load_estimator(model.model_run_name)
    y_pred = model.predict(X_test)

    match cfg.prediction.mode:
        case "predict":
            return y_pred
        case "evaluate":
            return y_pred, model.evaluate(X_test, y_test, cfg.loss_fn)
        case _:
            raise ValueError(f"Invalid mode: {cfg.prediction.mode}")