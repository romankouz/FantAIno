import FantAIno
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from utils.analytics import generate_confusion_matrix

root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))
melondy_and_spotify_df = pd.read_csv(os.path.join(root_dir, "data", "processed", "melondy_and_spotify.csv")).dropna()

pipe = Pipeline([
    # ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor())
])

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

pipe.fit(X=FantAIno_KNN_X_train, y=FantAIno_KNN_y_train)

raw_preds = pipe.predict(FantAIno_KNN_X_test)
preds = np.clip(np.rint(raw_preds), a_min=-1, a_max=10).astype(int)
acc = accuracy_score(y_true=FantAIno_KNN_y_test, y_pred=preds)

print(f"The Random Forest Regressor accuracy is {acc}")
mode = FantAIno_KNN_y_test.mode()[0]
print(f"The baseline accuracy is {np.mean(FantAIno_KNN_y_test.to_numpy() == mode)}")

generate_confusion_matrix(FantAIno_KNN_y_test, preds, "random_forest_regressor_CM")
