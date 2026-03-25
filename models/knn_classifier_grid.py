import FantAIno
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.analytics import generate_confusion_matrix

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
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
    "knn__n_neighbors": [2, 5, 10, 20, 30, 50, 100],
    "knn__weights": ["uniform", "distance"],
}
grid_search_cv = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="roc_auc_ovo_weighted")
best_model = grid_search_cv.fit(X=FantAIno_KNN_X_train, y=FantAIno_KNN_y_train)
# save all performance results
results_df = pd.DataFrame(grid_search_cv.cv_results_)
results_df.to_csv('results/knn_classification_cv_results.csv', index=False)

preds = best_model.predict(FantAIno_KNN_X_test)
acc = accuracy_score(y_true=FantAIno_KNN_y_test, y_pred=preds)

print(f"The best accuracy was {acc}")
mode = FantAIno_KNN_y_test.mode()[0]
print(f"The baseline accuracy is {np.mean(FantAIno_KNN_y_test.to_numpy() == mode)}")

generate_confusion_matrix(FantAIno_KNN_y_test, preds, "knn_classifier_grid_CM")
test_results = pd.concat([FantAIno_KNN_X_test, FantAIno_KNN_y_test, pd.Series(preds, index=FantAIno_KNN_y_test.index, name="prediction")], axis=1)
test_results.to_csv("results/test_songs.csv", index=False)

