import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayesDiscrete(BaseEstimator, ClassifierMixin):

    def __init__(self, domain_sizes, laplace, safe_computation=True):
        self.domain_sizes = domain_sizes
        self.laplace = laplace
        self.safe_computation = safe_computation

    def fit(self, X, y):
        m, n = X.shape  # m - no. of rows (data points), n - no. of features / attributes
        self.class_labels_ = np.unique(y)
        K = self.class_labels_.size  # no. of classes
        self.domain_sizes_ = self.domain_sizes
        self.PY_ = np.zeros(K)  # 1-d array with class probabilities (a priori)
        q = np.max(self.domain_sizes_)  # largest domain size
        self.P_ = np.zeros((K, n, q))  # 3-d array with conditional probabilities: P[y, j, v] = Prob.(X_j = v | Y = y)

        yy = np.zeros(m, dtype=np.int8)  # class labels mapped to indexes: 0, 1, 2...
        for index, label in enumerate(self.class_labels_):
            condition = y == label
            self.PY_[index] = np.mean(condition)
            yy[condition] = index

        for i in range(m):
            x = X[i]  # x = (3, 0, 2, 4, ...) y = 1
            for j in range(n):
                self.P_[yy[i], j, x[j]] += 1

        if not self.laplace:
            for index, label in enumerate(self.class_labels_):
                self.P_[index] /= self.PY_[index] * m
        else:
            for index, label in enumerate(self.class_labels_):
                for j in range(n):
                    self.P_[index, j] = (self.P_[index, j] + 1) / (self.PY_[index] * m + self.domain_sizes_[j])

        # Compute and memorize logarithms of probabilities for safe computation
        if self.safe_computation:
            self.P_ = np.log(self.P_)
            self.PY_ = np.log(self.PY_)

        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.class_labels_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        m, n = X.shape
        K = self.PY_.size
        log_probas = np.zeros((m, K))
        for i in range(m):
            x = X[i]  # e.g. x = (2, 0, 4, 1, 1, 0, ... )
            for k in range(K):
                log_prob = self.PY_[k]
                for j in range(n):
                    log_prob += self.P_[k, j, x[j]]
                log_probas[i, k] = log_prob

        # Convert back from logarithms to probabilities using log-sum-exp trick
        if self.safe_computation:
            max_log_probs = np.max(log_probas, axis=1, keepdims=True)
            probas = np.exp(log_probas - max_log_probs)
            norm_consts = np.sum(probas, axis=1, keepdims=True)
            probas /= norm_consts
        else:
            probas = np.exp(log_probas)

        return log_probas

        """
        When using safe computation, The difference in accuracy results can occur due to the inherent limitations of floating-point arithmetic.
        Exponentiating logarithmic values followed by normalization (log-sum-exp trick) and 
        directly exponentiating logarithmic values can lead to slightly different numerical results. 
        These differences can accumulate and affect the final probabilities, resulting in variations in accuracy.
        In most cases, the differences in accuracy between the two approaches should be minimal. 
        However, if your dataset has a large number of features or extremely small probabilities, 
        the differences in numerical computation can become more significant.
        """
