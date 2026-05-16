import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        n = len(y_true)
        epsilon = 1e-7
        pre = -1 * (1 / n)
        term = np.dot(y_true, np.log(y_pred + epsilon)) + np.dot((1 - y_true), np.log((1 - y_pred) + epsilon))
        return np.round(pre * term, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        n = len(y_true)
        pre = -1* (1/n)
        rows, cols = y_true.shape
        classes = cols
        loss = 0
        for i in range(rows): 
            for c in range(classes):
                product = y_true[i][c] *  np.log(y_pred[i][c] + 1e-7)
                loss += product
        return np.round(pre * loss, 4)