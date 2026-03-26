import FantAIno
import numpy as np
import os
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, Pool

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

catboost = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='MultiClass',
                           verbose=True)

catboost.fit(FantAIno_KNN_X_train, FantAIno_KNN_y_train)

preds = catboost.predict(FantAIno_KNN_X_test)
acc = accuracy_score(y_true=FantAIno_KNN_y_test, y_pred=preds)

print(f"The Catboost Classifier accuracy is {acc}")
mode = FantAIno_KNN_y_test.mode()[0]
print(f"The baseline accuracy is {np.mean(FantAIno_KNN_y_test.to_numpy() == mode)}")