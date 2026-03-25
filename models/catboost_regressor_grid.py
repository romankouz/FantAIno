import FantAIno
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

from utils.analytics import generate_confusion_matrix

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("catboost", CatBoostRegressor(verbose=False))
])

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

param_grid = {
    "catboost__iterations": [100, 250, 500],
    "catboost__depth": [4, 6, 8],
    "catboost__l2_leaf_reg": [0, 0.01, 0.1],
    "catboost__bagging_temperature": [0, 0.01, 0.1],
    "catboost__random_strength": [0, 0.01, 0.1],
}

param_grid = {
    "catboost__iterations": [100, 250],
    "catboost__depth": [4, 6],
    "catboost__l2_leaf_reg": [0, 0.01],
    "catboost__bagging_temperature": [0, 0.01],
    "catboost__random_strength": [0, 0.01],
}

grid_search_cv = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="roc_auc_ovo_weighted", verbose=1)
best_model = grid_search_cv.fit(X=FantAIno_KNN_X_train, y=FantAIno_KNN_y_train)

# save all performance results
results_df = pd.DataFrame(grid_search_cv.cv_results_)
results_df.to_csv('results/catboost_regression_cv_results.csv', index=False)

raw_preds = best_model.predict(FantAIno_KNN_X_test)
preds = np.clip(np.rint(raw_preds), a_min=-1, a_max=10).astype(int)
acc = accuracy_score(y_true=FantAIno_KNN_y_test, y_pred=preds)

print(f"The best accuracy was {acc}")
mode = FantAIno_KNN_y_test.mode()[0]
print(f"The baseline accuracy is {np.mean(FantAIno_KNN_y_test.to_numpy() == mode)}")

generate_confusion_matrix(FantAIno_KNN_y_test, preds, "catboost_regression_grid_CM")