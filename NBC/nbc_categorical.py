import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayesCategorical(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.class_labels_, y = np.unique(y, return_inverse=True)
        K = self.class_labels_.size

        self.PY_ = np.zeros(K)  # class probabilities (a priori)
        n_features = X.shape[1]
        max_feature_value = np.max(X)

        self.P_ = np.zeros((K, n_features, max_feature_value + 1))  # conditional probabilities

        for index, label in enumerate(self.class_labels_):
            condition = y == index
            self.PY_[index] = np.mean(condition)

        for i in range(X.shape[0]):
            x = X[i]
            for j in range(n_features):
                self.P_[y[i], j, x[j]] += 1

        for index, label in enumerate(self.class_labels_):
            self.P_[index] /= self.PY_[index] * X.shape[0]

        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.class_labels_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        m, n = X.shape
        K = self.PY_.size
        probas = np.ones((m, K))

        for i in range(m):
            x = X[i]
            for k in range(K):
                for j in range(n):
                    probas[i, k] *= self.P_[k, j, x[j]]
                probas[i, k] *= self.PY_[k]

        return probas
