# Implementation of NBC for Discrete and Categorical Datasets

## Introduction

The Naive Bayes Classifier is a probabilistic algorithm that leverages Bayes' theorem to classify data points into predefined categories or classes. It's particularly well-suited for text classification and other tasks where the assumption of independence between features holds reasonably well.

This project offers two implementations of NBC:

NaiveBayesCategorical (nbc_categorical.py): This class is designed for categorical datasets. It calculates class probabilities and conditional probabilities for each feature.

NaiveBayesDiscrete (nbc_discrete.py): This class is suitable for discrete datasets. It includes options for Laplace smoothing and safe computation of probabilities, which can be beneficial for certain datasets.

## Implementation

The project includes the following important code files:

- `nbc_categorical.py`: Implementation of the NaiveBayesCategorical class for categorical datasets.
- `nbc_discrete.py`: Implementation of the NaiveBayesDiscrete class for discrete datasets.

The `main_categorical.py` and `main_discrete.py` scripts demonstrate how to use these implementations for classification tasks. The code includes functions for data preprocessing, train-test splitting, and discretization where necessary.

## Usage

To use the Naive Bayes classifiers for your own datasets, you can follow these general steps:

1. Import the relevant NaiveBayes class (`NaiveBayesCategorical` or `NaiveBayesDiscrete`).

2. Load and preprocess your dataset, ensuring it's in the appropriate format.

3. Create an instance of the classifier and fit it to your training data.

4. Use the trained classifier to make predictions on test data.

5. Evaluate the classifier's performance using appropriate metrics.

## Datasets

This project includes sample datasets for testing:

- `agaricus-lepiota.data` (for categorical data)
- `waveform.data` (for discrete data)

You can replace these datasets with your own data by modifying the data loading code in the main scripts.

## Installation

To run the code in this project, you'll need Python and the following libraries:

- NumPy
- scikit-learn (for train-test splitting and performance evaluation)

You can install these dependencies using pip:

```bash
pip install numpy scikit-learn
```
Once you have the required dependencies, you can run the main scripts (main_categorical.py and main_discrete.py) to perform classification tasks.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to customize and extend the code to suit your specific needs. If you have any questions or encounter issues, please don't hesitate to reach out.

Happy coding!
