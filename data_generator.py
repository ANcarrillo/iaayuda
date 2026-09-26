"""
data_generator.py — Genera datos 3D con distribución normal por clase.
"""
import numpy as np


def generate_data(n_samples, mean, std, n_classes=2, separable=True,
                   samples_per_class=None):
    X_list, y_list = [], []
    for c in range(n_classes):
        count = (samples_per_class[c]
                  if samples_per_class and c < len(samples_per_class) and samples_per_class[c] > 0
                  else n_samples)
        offset = c * std * 4.0 if separable else c * std * 0.4
        center = np.array([mean + offset, mean + offset, mean + offset])
        samples = np.random.normal(loc=center, scale=std, size=(count, 3))
        X_list.append(samples)
        y_list.extend([c] * count)
    X = np.vstack(X_list)
    y = np.array(y_list)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]
