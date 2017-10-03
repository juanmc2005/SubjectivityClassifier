import pandas as pd
from classifiers import NNClassifier


def load_data():
    x_train = pd.read_csv('./nn/x_train.csv', delimiter=',').values
    y_train = pd.read_csv('./nn/y_train.csv', delimiter=',').values.flatten()
    x_test = pd.read_csv('./nn/x_test.csv', delimiter=',').values
    y_test = pd.read_csv('./nn/y_test.csv', delimiter=',').values.flatten()
    return x_train, y_train, x_test, y_test


def load_data_into_classifier():
    x_train, y_train, x_test, y_test = load_data()
    return NNClassifier(x_train, y_train, x_test, y_test)
