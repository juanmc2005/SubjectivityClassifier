import pandas as pd
from classifiers import SVMClassifier


def load_data():
    x_train = pd.read_csv('./svm_cross/x_train.csv', delimiter=',').values
    y_train = pd.read_csv('./svm_cross/y_train.csv', delimiter=',').values.flatten()
    x_test = pd.read_csv('./svm_cross/x_test.csv', delimiter=',').values
    y_test = pd.read_csv('./svm_cross/y_test.csv', delimiter=',').values.flatten()
    return x_train, y_train, x_test, y_test


def load_data_into_classifier():
    x_train, y_train, x_test, y_test = load_data()
    return SVMClassifier(x_train, y_train, x_test, y_test)
