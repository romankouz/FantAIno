"""
The official training script for FantAIno.
"""

from hydra.utils import instantiate
from omegaconf import DictConfig

def train(cfg: DictConfig) -> None:
    """Train a FantAIno model."""

    album_data_preprocessing_pipeline = instantiate(cfg.preprocessing.album_data_preprocessing_pipeline)
    lyrics_preprocessing_pipeline = instantiate(cfg.preprocessing.lyrics_preprocessing_pipeline)
    FantAIno_df_obj = instantiate(
        cfg.dataset,
        album_data_preprocessing_pipeline=album_data_preprocessing_pipeline,
        lyrics_preprocessing_pipeline=lyrics_preprocessing_pipeline
    )

    X_train, y_train = FantAIno_df_obj.FantAIno_df_X, FantAIno_df_obj.FantAIno_df_y

    model = instantiate(cfg.model, _convert_="partial")
    model.train_model(X_train, y_train)
    model.save_estimator(model.model_run_name)
