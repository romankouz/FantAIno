import FantAIno
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

root_dir = os.path.dirname(os.path.abspath(FantAIno.__path__[0]))
melondy_and_spotify_df = pd.read_csv(os.path.join(root_dir, "data", "processed", "melondy_and_spotify.csv")).dropna()

FantAIno_KNN_features = [
    "total_tracks",
    "release_year",
    "release_month",
    "album_duration_in_s",
    "explicit_proportion",
    "num_features"
]
FantAIno_KNN_response = melondy_and_spotify_df["rating"]
FantAIno_KNN_df = melondy_and_spotify_df.drop(columns=["rating"])

(
    melondy_and_spotify_df_X_train,
    melondy_and_spotify_df_X_test,
    FantAIno_KNN_y_train,
    FantAIno_KNN_y_test
) = train_test_split(FantAIno_KNN_df, FantAIno_KNN_response, stratify=FantAIno_KNN_response)

FantAIno_KNN_X_train = melondy_and_spotify_df_X_train[FantAIno_KNN_features]
FantAIno_KNN_X_test = melondy_and_spotify_df_X_test[FantAIno_KNN_features]

param_grid = {
    "knn__n_neighbors": [2, 5, 10, 20, 30, 50, 100],
    "knn__weights": ["uniform", "distance"],
}
grid_search_cv = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring="roc_auc_ovo_weighted")
best_model = grid_search_cv.fit(X=FantAIno_KNN_X_train, y=FantAIno_KNN_y_train)
# save all performance results
results_df = pd.DataFrame(grid_search_cv.cv_results_)
results_df.to_csv('results/knn_classification_cv_results.csv', index=False)

# Get the unique labels from the actual test data
labels = sorted(FantAIno_KNN_y_test.unique())

preds = best_model.predict(FantAIno_KNN_X_test)
acc = accuracy_score(y_true=FantAIno_KNN_y_test, y_pred=preds)
cm = confusion_matrix(y_true=FantAIno_KNN_y_test, y_pred=preds, labels=labels)

print(f"The best accuracy was {acc}")
mode = FantAIno_KNN_y_test.mode()[0]
print(f"The baseline accuracy is {np.mean(FantAIno_KNN_y_test.to_numpy() == mode)}")

# Create a heatmap for the confusion matrix
sns.heatmap(cm,
            annot=True,  # Show the numbers in each cell
            fmt='g',     # Format the numbers as general (non-scientific)
            xticklabels=labels,  # Set labels for the x-axis (predictions)
            yticklabels=labels)  # Set labels for the y-axis (actuals)

# Set the label for the y-axis
plt.ylabel('Actual', fontsize=13)
# Set the title of the plot
plt.title('Confusion Matrix', fontsize=17, pad=20)
# Position the x-axis label at the top
plt.gca().xaxis.set_label_position('top')
# Set the label for the x-axis
plt.xlabel('Prediction', fontsize=13)
# Move the x-axis ticks to the top
plt.gca().xaxis.tick_top()

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)

plt.savefig("results/knn_classification_cv_CM.png")
test_results = pd.concat([melondy_and_spotify_df_X_test, FantAIno_KNN_y_test, pd.Series(preds, index=FantAIno_KNN_y_test.index, name="prediction")], axis=1)
test_results.to_csv("results/test_songs.csv", index=False)

