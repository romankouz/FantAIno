import FantAIno

from omegaconf import DictConfig, OmegaConf
import hydra

from models.knn_classifier_extend import KNNClassifier


# temporary imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

model = KNNClassifier(model_run_name="master")
model.train(FantAIno_KNN_X_train, FantAIno_KNN_y_train)
model.save_estimator(model.model_run_name)

# @hydra.main(version_base=None)
# def my_app(cfg: DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))

# if __name__ == "__main__":
#     my_app()

# prepare the data

# create the model

# train the model

