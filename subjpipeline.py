from optimizers import SVMOptimizer, NNOptimizer
from nn_cross.data import load_data_into_classifier as load_nn
from svm_cross.data import load_data_into_classifier as load_svm


class Pipeline:

    def __init__(self, preprocessor):
        self.sentences = None
        self.processed_sentences = None
        self.labels = None
        self.matrices = None
        self.vectors = None
        self.classifier = None
        self.preprocessor = preprocessor
        self.svm_optimizer = None
        self.nn_optimizer = None

    def preprocess(self, verbose=True):
        self.sentences, self.processed_sentences, self.labels, self.matrices, self.vectors =\
            self.preprocessor.preprocess(verbose)
        self.svm_optimizer = SVMOptimizer(self.vectors, self.labels)
        self.nn_optimizer = NNOptimizer(self.vectors, self.labels)
        return self

    def optimal_svm(self, results_file='svm_results.csv', verbose=True):
        self.classifier, config = self.svm_optimizer.optimal(results_file, verbose)
        return config

    def optimal_nn(self, results_file='nn_results.csv', verbose=True):
        self.classifier, config = self.nn_optimizer.optimal(results_file, verbose)
        return config

    def load_svm(self):
        self.classifier = load_svm().configure('sigmoid', 0.01, 0.001).fit()
        return self

    def load_nn(self):
        self.classifier = load_nn().configure('adam', 'relu', 0.01, (3,)).fit()
        return self

    def predict(self, docname, verbose=True):
        vectors, estimated_labels = self.preprocessor.production_preprocess(docname, verbose)
        return vectors, self.classifier.predict(vectors), estimated_labels
