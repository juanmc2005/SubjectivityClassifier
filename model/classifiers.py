from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as scores


class SubjectivityClassifier:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = None

    def _build_classifier(self):
        raise NotImplementedError()

    def fit(self):
        self.classifier = self._build_classifier()
        self.classifier.fit(self.x_train, self.y_train)
        return self

    def evaluate(self):
        predicted = self.classifier.predict(self.x_test)
        precision, recall, fscore, support = scores(self.y_test, predicted)
        return precision, recall, fscore


class SVMClassifier(SubjectivityClassifier):

    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.kernel = 'linear'
        self.C = 1
        self.gamma = 'auto'

    def configure(self, kernel='linear', c=1, gamma='auto'):
        self.kernel = kernel
        self.C = c
        self.gamma = gamma
        return self

    def _build_classifier(self):
        return SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)


class NNClassifier(SubjectivityClassifier):

    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)

    def _build_classifier(self):
        return MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(15, 8, 4), random_state=1)


class SVMConfig:

    def __init__(self, x_train, y_train, x_test, y_test, kernel, precision, recall, fscore, c, gamma):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.kernel = kernel
        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.c = c
        self.gamma = gamma

    def __str__(self):
        return """
            kernel: {}
            precision: {}
            recall: {}
            f-score: {}
            c: {}
            gamma: {}
        """.format(self.kernel, self.precision, self.recall, self.fscore, self.c, self.gamma)
