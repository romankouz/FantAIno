from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import FantAIno

# predict using a trainedmodel
def predict(cfg: DictConfig) -> None:

    # temporary imports
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # prepare the data
    root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))
    melondy_and_spotify_df = pd.read_csv(os.path.join(root_dir, "data", "processed", "melondy_and_spotify.csv")).dropna()

    DROPPED_FEATURES = [
        "artist",
        "album",
        "image_url",
        "featured_artists",
        "track_names",
    ]

    FantAIno_KNN_response = melondy_and_spotify_df["rating"]
    FantAIno_KNN_df = melondy_and_spotify_df.drop(["rating"] + DROPPED_FEATURES, axis=1)

    (
        FantAIno_KNN_X_train,
        FantAIno_KNN_X_test,
        FantAIno_KNN_y_train,
        FantAIno_KNN_y_test
    ) = train_test_split(FantAIno_KNN_df, FantAIno_KNN_response, stratify=FantAIno_KNN_response)

    model = instantiate(cfg.model, _convert_="partial")
    model.load_estimator(model.model_run_name)
    predictions = model.predict(FantAIno_KNN_X_test)
    if cfg.mode == "evaluate":
        loss = model.evaluate(FantAIno_KNN_X_test, FantAIno_KNN_y_test, cfg.loss_fn)
        return predictions, loss
    else:
        return predictions