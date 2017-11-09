from itertools import product
from classifiers import SVMClassifier, NNClassifier, SVMConfig, NNConfig
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from verbose import vlist, vprint


class Optimizer:

    @staticmethod
    def max_config(results):
        max_config = None
        for res in results:
            if max_config is None or res.cross_mean_fscore > max_config.cross_mean_fscore:
                max_config = res
        return max_config

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def _classifier(self, x_train, y_train, x_test, y_test):
        raise NotImplementedError()

    def _split_data(self):
        return train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)

    def _new_classifier(self):
        x_train, x_test, y_train, y_test = self._split_data()
        return self._classifier(x_train, y_train, x_test, y_test)

    def optimal(self, results_file, verbose):
        raise NotImplementedError()


class SVMOptimizer(Optimizer):

    def __init__(self, X, y):
        super().__init__(X, y)

    def _classifier(self, x_train, y_train, x_test, y_test):
        return SVMClassifier(x_train, y_train, x_test, y_test)

    def build_config(self, clf, kernel, c, gamma):
        precision, recall, fscore = clf.configure(kernel, c, gamma).fit().evaluate()
        mean_fscore = cross_val_score(clf.classifier, self.X, self.y, cv=5, scoring='f1_macro').mean()
        return SVMConfig(clf.x_train, clf.y_train, clf.x_test, clf.y_test,
                         precision, recall, fscore, mean_fscore, kernel, c, gamma)

    def optimal(self, results_file, verbose):
        cs = [0.01, 0.1, 1, 10, 100, 1000]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        cs_gammas = list(product(cs, gammas))
        results = []
        with open(results_file, 'a', encoding='utf8') as f:
            f.write('ITER,KERNEL,PRECISION,RECALL,F-SCORE,CROSS-F-SCORE,C,GAMMA\n')
            for i in vlist(range(100), 'Finding best SVM config', verbose):
                classifier = self._new_classifier()
                for kernel in ['linear', 'sigmoid', 'rbf']:
                    if kernel != 'linear':
                        for c, gamma in cs_gammas:
                            config = self.build_config(classifier, kernel, c, gamma)
                            results.append(config)
                            f.write(str(i) + ',' + config.csv_str() + '\n')
                    else:
                        for c in cs:
                            config = self.build_config(classifier, kernel, c, 'auto')
                            results.append(config)
                            f.write(str(i) + ',' + config.csv_str() + '\n')
        max_config = Optimizer.max_config(results)
        return max_config.trained_classifier(), max_config


class NNOptimizer(Optimizer):

    @staticmethod
    def layers():
        layer_sizes = list(range(2, 7))
        res = [(1,), (2,), (3,), (4,), (5,), (6,)]
        res.extend(list(product(layer_sizes, layer_sizes)))
        res.extend(list(product(layer_sizes, layer_sizes, layer_sizes)))
        return res

    def __init__(self, X, y):
        super().__init__(X, y)

    def _classifier(self, x_train, y_train, x_test, y_test):
        return NNClassifier(x_train, y_train, x_test, y_test)

    def build_config(self, clf, solver, activation, alpha, hlayer):
        precision, recall, fscore = clf.configure(solver, activation, alpha, hlayer).fit().evaluate()
        mean_fscore = cross_val_score(clf.classifier, self.X, self.y, cv=5, scoring='f1_macro').mean()
        return NNConfig(clf.x_train, clf.y_train, clf.x_test, clf.y_test,
                        precision, recall, fscore, mean_fscore, solver, activation, alpha, hlayer)

    def optimal(self, results_file, verbose):
        solvers = ['lbfgs', 'adam']
        activations = ['logistic', 'tanh', 'relu']
        alphas = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.2]
        params = list(product(solvers, activations, alphas, NNOptimizer.layers()))
        results = []
        vprint('Finding best NN config...', verbose)
        with open(results_file, 'a', encoding='utf8') as f:
            f.write('ITER,PRECISION,RECALL,F-SCORE,CROSS-F-SCORE,SOLVER,ACTIVATION,ALPHA,HIDDEN LAYERS\n')
            for i in range(5):
                classifier = self._new_classifier()
                for solver, activation, alpha, hlayer in vlist(params, 'Training iteration {}'.format(i), verbose):
                    config = self.build_config(classifier, solver, activation, alpha, hlayer)
                    results.append(config)
                    f.write(str(i) + ',' + config.csv_str() + '\n')
        max_config = Optimizer.max_config(results)
        classifier = max_config.trained_classifier()
        return classifier, max_config
