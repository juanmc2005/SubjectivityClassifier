from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as scores
import pathlib
from verbose import vprint


class SingleSubjectivityClassifier:

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

    def predict(self, X):
        return self.classifier.predict(X)


class SVMClassifier(SingleSubjectivityClassifier):

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


class NNClassifier(SingleSubjectivityClassifier):

    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.solver = 'adam'
        self.activation = 'tanh'
        self.alpha = 1e-4
        self.hidden_layer_sizes = (int((len(x_train[0]) + 1) / 2),)

    def configure(self, solver, activation, alpha, hidden_layer_sizes):
        self.solver = solver
        self.activation = activation
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        return self

    def _build_classifier(self):
        return MLPClassifier(solver=self.solver,
                             alpha=self.alpha,
                             activation=self.activation,
                             hidden_layer_sizes=self.hidden_layer_sizes)


class Config:

    def __init__(self, x_train, y_train, x_test, y_test, precision, recall, fscore, cross_mean_fscore):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.cross_mean_fscore = cross_mean_fscore

    def dump(self, verbose=True):
        vprint('Dumping configuration...', verbose)
        dir = self.dump_dir()
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        with open(dir + '/x_train.csv', 'a', encoding='utf8') as f:
            f.write('MAX SWF-ISF,AVG FRS,AVG FRO,FR FRS/FRO,FRM,PABS,PATS\n')
            for v in self.x_train:
                f.write(str(v).replace('[', '').replace(']', '') + '\n')
        with open(dir + '/y_train.csv', 'a', encoding='utf8') as f:
            f.write('Subjective\n')
            for v in self.y_train:
                f.write(str(v).replace('[', '').replace(']', '') + '\n')
        with open(dir + '/x_test.csv', 'a', encoding='utf8') as f:
            f.write('MAX SWF-ISF,AVG FRS,AVG FRO,FR FRS/FRO,FRM,PABS,PATS\n')
            for v in self.x_test:
                f.write(str(v).replace('[', '').replace(']', '') + '\n')
        with open(dir + '/y_test.csv', 'a', encoding='utf8') as f:
            f.write('Subjective\n')
            for v in self.y_test:
                f.write(str(v).replace('[', '').replace(']', '') + '\n')
        with open(dir + '/stats.txt', 'a', encoding='utf8') as f:
            f.write(str(self))
        vprint('Done', verbose)

    def trained_classifier(self):
        raise NotImplementedError()

    def csv_str(self):
        raise NotImplementedError()

    def dump_dir(self):
        raise NotImplementedError()


class SVMConfig(Config):

    def __init__(self, x_train, y_train, x_test, y_test, precision,
                 recall, fscore, cross_mean_fscore, kernel, c, gamma):
        super().__init__(x_train, y_train, x_test, y_test, precision, recall, fscore, cross_mean_fscore)
        self.kernel = kernel
        self.c = c
        self.gamma = gamma

    def trained_classifier(self):
        return SVMClassifier(self.x_train, self.y_train, self.x_test, self.y_test)\
            .configure(self.kernel, self.c, self.gamma)\
            .fit()

    def __str__(self):
        return """
            kernel: {}
            precision: {}
            recall: {}
            f-score: {}
            cross validation f-score: {}
            c: {}
            gamma: {}
        """.format(self.kernel, self.precision, self.recall, self.fscore, self.cross_mean_fscore, self.c, self.gamma)

    def csv_str(self):
        return "{},{},{},{},{},{},{}".format(self.kernel, self.precision, self.recall,
                                             self.fscore, self.cross_mean_fscore, self.c, self.gamma)

    def dump_dir(self):
        return './svm_dump'


class NNConfig(Config):

    def __init__(self, x_train, y_train, x_test, y_test, precision, recall,
                 fscore, cross_mean_fscore, solver, activation, alpha, hlayers):
        super().__init__(x_train, y_train, x_test, y_test, precision, recall, fscore, cross_mean_fscore)
        self.solver = solver
        self.activation = activation
        self.alpha = alpha
        self.hlayers = hlayers

    def trained_classifier(self):
        return NNClassifier(self.x_train, self.y_train, self.x_test, self.y_test)\
            .configure(self.solver, self.activation, self.alpha, self.hlayers)\
            .fit()

    def __str__(self):
        return """
            precision: {}
            recall: {}
            f-score: {}
            cross validation f-score: {}
            solver: {}
            activation function: {}
            alpha: {}
            hidden layers: {}
        """.format(self.precision, self.recall, self.fscore, self.cross_mean_fscore,
                   self.solver, self.activation, self.alpha, self.hlayers)

    def csv_str(self):
        return "{},{},{},{},{},{},{},{}"\
            .format(self.precision, self.recall, self.fscore, self.cross_mean_fscore,
                    self.solver, self.activation, self.alpha, self.hlayers)

    def dump_dir(self):
        return './nn_dump'
