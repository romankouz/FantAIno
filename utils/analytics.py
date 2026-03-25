import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix

def generate_confusion_matrix(y_true, y_pred, model_filename):
    """
    Generates a confusion matrix from the true and predicted labels.
    """
    
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
    """
        Saves song name, artist, true rating, and predicted rating.
    """