from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as scores


class Pipeline:

    def __init__(self):
        self.sentences = None
        self.processed_sentences = None
        self.labels = None
        self.matrices = None
        self.vectors = None
        self.classifier = None

    def fit(self, kernel, c, gamma):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


class SubjectivityPipeline(Pipeline):

    def __init__(self, preprocessor):
        Pipeline.__init__(self)
        self.preprocessor = preprocessor

    def preprocess(self, verbose=True):
        self.sentences, self.processed_sentences, self.labels, self.matrices, self.vectors =\
            self.preprocessor.preprocess(verbose)
        return self

    def build_classifier(self):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.vectors, self.labels, test_size=0.2, stratify=self.labels)
        self.classifier = SubjClassifier(x_train, y_train, x_test, y_test)
        return self

    def fit(self, kernel, c=1, gamma='auto'):
        self.classifier.fit(kernel, c, gamma)
        return self

    def evaluate(self):
        return self.classifier.evaluate()

    # TODO override methods


class SubjClassifier:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = None

    def fit(self, kernel, c, gamma):
        self.classifier = SVC(kernel=kernel, C=c, gamma=gamma, degree=2)
        self.classifier.fit(self.x_train, self.y_train)
        return self

    def evaluate(self):
        predicted = self.classifier.predict(self.x_test)
        precision, recall, fscore, support = scores(self.y_test, predicted)
        return precision, recall, fscore
