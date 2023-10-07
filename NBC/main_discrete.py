import numpy as np
from nbc_discrete import NaiveBayesDiscrete


def train_test_split(X, y, train_ratio=0.75, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    indexes = np.random.permutation(m)
    X = X[indexes]
    y = y[indexes]
    i = int(np.round(train_ratio * m))
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:]
    y_test = y[i:]
    return X_train, y_train, X_test, y_test

# It can handle both missing values and division by 0
def discretize_data(X, n_bins=5, mins=None, maxes=None):
    if mins is None:
        mins = np.nanmin(X, axis=0)
        maxes = np.nanmax(X, axis=0)

    # Calculate column means while ignoring missing values ('?')
    means = np.nanmean(X, axis=0)

    # Replace missing values ('?') with the respective attribute mean
    X = np.where(X == '?', means, X)

    # Convert string values to float
    X = X.astype(float)

    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    X_modified = X + epsilon

    X_d = np.clip(((X_modified - mins) / (maxes - mins) * n_bins).astype(np.int8), 0, n_bins - 1)
    return X_d, mins, maxes


if __name__ == '__main__':

    print("STARTING...")
    n_bins = 5
    # D = np.genfromtxt("wine.data", delimiter=",")
    D = np.genfromtxt("waveform.data", delimiter=",")
    # D = np.genfromtxt("polish_bank.data", delimiter=",")
    y = D[:, -1].astype(np.int8)
    X = D[:, 0:-1]
    n = X.shape[1]
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_ratio=0.75, seed=2)
    X_train_d, mins_ref, maxes_ref = discretize_data(X_train, n_bins)
    X_test_d, _, _ = discretize_data(X_test, n_bins, mins_ref, maxes_ref)

    domain_sizes_wine = n_bins * np.ones(n, dtype=np.int8)
    clf = NaiveBayesDiscrete(domain_sizes=domain_sizes_wine, laplace=True, safe_computation=True)
    clf.fit(X_train_d, y_train)
    print(clf.PY_)

    # predictions = clf.predict(X_test_d)
    # print(predictions)
    # print(y_test)
    acc_test = clf.score(X_test_d, y_test)  # np.mean(predictions == y_test)
    print(f"ACC TEST: {acc_test}")
    acc_train = clf.score(X_train_d, y_train)  # np.mean(predictions == y_test)
    print(f"ACC TRAIN: {acc_train}")


    print("ALL DONE.")
