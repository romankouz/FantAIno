import numpy as np
from sklearn.metrics import accuracy_score

def rounded_regression_accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, np.round(y_pred))

def cosine_similarity(a, b) -> float:
    return a.dot(b.T).item()