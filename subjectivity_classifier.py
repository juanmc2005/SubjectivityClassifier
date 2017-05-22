

class SubjClassifier:

    def __init__(self, sentences=None):
        if sentences is not None: self._sentences = []
        else: self._sentences = sentences

    def preprocess(self):
        # TODO preprocess sentences
        pass

    def predict(self):
        # TODO actually predict sentences using model
        return self._sentences
