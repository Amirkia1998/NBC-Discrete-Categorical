import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nbc_categorical import NaiveBayesCategorical


def preprocess_data(X, y):
    # Convert the feature values to numerical format
    X = np.array([[ord(value) for value in row] for row in X])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Load the dataset
    # data = np.loadtxt("kr-vs-kp.data", dtype=str, delimiter=",")
    data = np.loadtxt("agaricus-lepiota.data", dtype=str, delimiter=",")


    # Split the data into features (X) and labels (y)
    X = data[:, :-1]
    y = data[:, -1]

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Create an instance of the NaiveBayesChess classifier
    nb_classifier = NaiveBayesCategorical()

    # Fit the classifier on the training data
    nb_classifier.fit(X_train, y_train)

    # Predict the labels for testing data
    y_test_pred = nb_classifier.predict(X_test)

    # Predict the labels for training data
    y_train_pred = nb_classifier.predict(X_train)

    # Calculate and print the accuracy scores
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print("Test Accuracy:", test_accuracy)
    print("Train Accuracy:", train_accuracy)
