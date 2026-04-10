import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Literal

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

from FantAIno.constants import RESULTS_DIR

def generate_confusion_matrix(y_true, y_pred, model_filename):
    """Generates a confusion matrix from the true and predicted labels."""
    cm = confusion_matrix(y_true, y_pred)
    labels = np.arange(-1, 11)
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=labels,
                yticklabels=labels)

    plt.ylabel('Actual', fontsize=13)
    plt.title(f'Confusion Matrix: Acc = {round(accuracy_score(y_true, y_pred), 2)}', fontsize=14, pad=10)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(top=0.75)
    plt.tight_layout()
    plt.savefig(f"results/{model_filename}.png")

def simple_results_csv(y_true, y_pred, output_filename):
    """Saves song name, artist, true rating, and predicted rating."""
    raise NotImplementedError("Not implemented yet.")

def refresh_results(
    classification_metric = accuracy_score,
    regression_metric = mean_squared_error
) -> pd.DataFrame:

    ### TEMPORARY CODE ###
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import FantAIno

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
    ) = train_test_split(FantAIno_KNN_df, FantAIno_KNN_response, stratify=FantAIno_KNN_response, random_state=888)

    ### TEMPORARY CODE ###

    results_dict = {}
    for root, dirs, files in os.walk(RESULTS_DIR):
        if dirs == []:
            for file in files:
                if file.endswith(".joblib"):
                    try:
                        model_name = os.path.basename(root)
                        run_name = os.path.splitext(file)[0]
                        trained_model = joblib.load(os.path.join(root, file))
                        preds = trained_model.predict(FantAIno_KNN_X_test)
                        if any(term in model_name for term in ["Ordinal Logistic", "Classifier"]):
                            metric = "accuarcy"
                            score = accuracy_score(FantAIno_KNN_y_test, preds)
                        else:
                            metric = "MSE"
                            score = mean_squared_error(FantAIno_KNN_y_test, preds)

                        results_dict[model_name] = [
                            run_name,
                            metric,
                            score,
                        ]
                    except Exception as e:
                        print(f"Model {model_name} failed to load with error: {e}")

    results_df = pd.DataFrame.from_dict(
        results_dict,
        orient="index",
        columns=["run_name", "metric", "score"]
    )

    results_df.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=True)

    return results_df