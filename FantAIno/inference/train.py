"""
The official training script for FantAIno.
"""

from hydra.utils import instantiate
from omegaconf import DictConfig

def train(cfg: DictConfig) -> None:
    """Train a FantAIno model."""

    FantAIno_df_obj = instantiate(cfg.dataset)

    X_train, y_train = FantAIno_df_obj.FantAIno_df_X_train, FantAIno_df_obj.FantAIno_df_y_train
    
    model = instantiate(cfg.model, _convert_="partial")
    model.train(X_train, y_train)
    model.save_estimator(model.model_run_name)
