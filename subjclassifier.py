class SubjClassifier:

    def __init__(self):
        self.sentences = None
        self.processed_sentences = None
        self.labels = None
        self.matrices = None

    def fit(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


class PreprocessorSubjClassifier(SubjClassifier):

    def __init__(self, preprocessor):
        SubjClassifier.__init__(self)
        self.preprocessor = preprocessor

    def preprocess(self, verbose=True):
        self.sentences, self.processed_sentences, self.labels, self.matrices = self.preprocessor.preprocess(verbose)
        return self

    # TODO override methods
