"""
perceptron.py — Perceptrón simple y One-vs-Rest multiclase.
"""
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.weight_history = []
        self.bias_history = []
        self.error_history = []

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.weight_history = []
        self.bias_history = []
        self.error_history = []
        for _ in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self._activate(xi)
                error = int(yi) - int(pred)
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            self.error_history.append(errors)
            if errors == 0:
                break
        return self

    def _activate(self, x):
        return 1 if np.dot(self.weights, x) + self.bias >= 0 else -1

    def predict_single(self, x):
        return self._activate(x)

    def predict(self, X):
        return np.array([self._activate(xi) for xi in X])


class OneVsRestPerceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.perceptrons = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            p = Perceptron(self.lr, self.max_epochs)
            p.fit(X, np.where(y == c, 1, -1))
            self.perceptrons[c] = p
        return self

    def predict_single(self, x):
        scores = {c: float(np.dot(self.perceptrons[c].weights, x) + self.perceptrons[c].bias)
                  for c in self.classes}
        return int(max(scores, key=scores.get))

    def predict(self, X):
        scores = np.array([[np.dot(self.perceptrons[c].weights, xi) + self.perceptrons[c].bias
                             for c in self.classes] for xi in X])
        return self.classes[np.argmax(scores, axis=1)]
