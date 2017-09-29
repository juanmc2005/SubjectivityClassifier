import pandas as pd
from classifiers import SVMClassifier


def load_data():
    x_train = pd.read_csv('./svm/x_train.csv', delimiter=',').values
    y_train = pd.read_csv('./svm/y_train.csv', delimiter=',').values.flatten()
    x_test = pd.read_csv('./svm/x_test.csv', delimiter=',').values
    y_test = pd.read_csv('./svm/y_test.csv', delimiter=',').values.flatten()
    print(y_train[0])
    return x_train, y_train, x_test, y_test


def load_data_into_classifier():
    x_train, y_train, x_test, y_test = load_data()
    return SVMClassifier(x_train, y_train, x_test, y_test)