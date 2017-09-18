from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as scores
from itertools import product
from tqdm import tqdm


def _verbose_iterable(iter, desc, verbose):
    if verbose:
        return tqdm(iter, desc)
    else:
        return iter


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


class Pipeline:

    def __init__(self, preprocessor):
        self.sentences = None
        self.processed_sentences = None
        self.labels = None
        self.matrices = None
        self.vectors = None
        self.classifier = None
        self.preprocessor = preprocessor

    def _split_data(self):
        return train_test_split(self.vectors, self.labels, test_size=0.2, stratify=self.labels)

    def _new_classifier(self):
        x_train, x_test, y_train, y_test = self._split_data()
        return SubjClassifier(x_train, y_train, x_test, y_test)

    def build_classifier(self):
        self.classifier = self._new_classifier()
        return self

    def preprocess(self, verbose=True):
        self.sentences, self.processed_sentences, self.labels, self.matrices, self.vectors =\
            self.preprocessor.preprocess(verbose)
        return self

    def optimal_svm(self, results_file='svm_results.csv', verbose=True):
        cs = [0.01, 0.1, 1, 10, 100, 1000]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        cs_gammas = list(product(cs, gammas))
        results = []
        with open(results_file, 'a', encoding='utf8') as f:
            f.write('ITER,KERNEL,PRECISION,RECALL,F-SCORE,C,GAMMA\n')
            for i in _verbose_iterable(range(100), 'Finding best SVM config', verbose):
                classifier = self._new_classifier()
                for kernel in ['linear', 'sigmoid', 'rbf']:
                    if kernel != 'linear':
                        for c, gamma in cs_gammas:
                            precision, recall, fscore = classifier.fit(kernel, c, gamma).evaluate()
                            results.append(SVMConfig(
                                classifier.x_train,
                                classifier.y_train,
                                classifier.x_test,
                                classifier.y_test,
                                kernel, precision, recall, fscore, c, gamma
                            ))
                            f.write(str(i) + ',' + kernel + ',' + str(precision) + ',' +
                                    str(recall) + ',' + str(fscore) + ',' + str(c) + ',' + str(gamma) + '\n')
                    else:
                        for c in cs:
                            precision, recall, fscore = classifier.fit(kernel, c).evaluate()
                            results.append(SVMConfig(
                                classifier.x_train,
                                classifier.y_train,
                                classifier.x_test,
                                classifier.y_test,
                                kernel, precision, recall, fscore, c, 'auto'
                            ))
                            f.write(str(i) + ',' + kernel + ',' + str(precision) + ',' +
                                    str(recall) + ',' + str(fscore) + ',' + str(c) + ',auto\n')
        max_config = None
        for res in results:
            if max_config is None or (res.fscore[0] > max_config.fscore[0] and res.fscore[1] > max_config.fscore[1]):
                max_config = res

        self.classifier = SubjClassifier(max_config.x_train, max_config.y_train, max_config.x_test, max_config.y_test)\
            .fit(max_config.kernel, max_config.c, max_config.gamma)
        return max_config


class SubjClassifier:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = None

    def fit(self, kernel, c=1, gamma='auto'):
        self.classifier = SVC(kernel=kernel, C=c, gamma=gamma)
        self.classifier.fit(self.x_train, self.y_train)
        return self

    def evaluate(self):
        predicted = self.classifier.predict(self.x_test)
        precision, recall, fscore, support = scores(self.y_test, predicted)
        return precision, recall, fscore
