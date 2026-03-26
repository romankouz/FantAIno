import FantAIno
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.analytics import generate_confusion_matrix


root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))
melondy_and_spotify_df = pd.read_csv(os.path.join(root_dir, "data", "processed", "melondy_and_spotify.csv")).dropna()

pipe = Pipeline([
    ("scaler", StandardScaler()),
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

param_grid = {
    "rf__n_estimators": [25, 50, 100, 250, 500, 1000],
    "rf__min_samples_leaf": [1, 2, 3, 5, 10, 25],
    "rf__max_features": [1.0, "sqrt", "log2", 0.75, 0.6, 0.2],
    "rf__min_impurity_decrease": [0.0, 0.001, 0.01, 0.1],
    "rf__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
}
grid_search_cv = GridSearchCV(estimator=pipe, param_grid=param_grid)
best_model = grid_search_cv.fit(X=FantAIno_KNN_X_train, y=FantAIno_KNN_y_train)

# save all performance results
results_df = pd.DataFrame(grid_search_cv.cv_results_)
results_df.to_csv('results/knn_regression_cv_results.csv', index=False)

raw_preds = pipe.predict(FantAIno_KNN_X_test)
preds = np.clip(np.rint(raw_preds), a_min=-1, a_max=10).astype(int)
acc = accuracy_score(y_true=FantAIno_KNN_y_test, y_pred=preds)

print(f"The best accuracy was {acc}")
mode = FantAIno_KNN_y_test.mode()[0]
print(f"The baseline accuracy is {np.mean(FantAIno_KNN_y_test.to_numpy() == mode)}")

generate_confusion_matrix(FantAIno_KNN_y_test, preds, "knn_classifier_grid_CM")
