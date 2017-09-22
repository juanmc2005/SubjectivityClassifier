from itertools import product
from classifiers import SVMClassifier, SVMConfig
from sklearn.model_selection import train_test_split
from verbose import vlist


class SVMOptimizer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def _split_data(self):
        return train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)

    def _new_classifier(self):
        x_train, x_test, y_train, y_test = self._split_data()
        return SVMClassifier(x_train, y_train, x_test, y_test)

    def optimal(self, results_file, verbose):
        cs = [0.01, 0.1, 1, 10, 100, 1000]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        cs_gammas = list(product(cs, gammas))
        results = []
        with open(results_file, 'a', encoding='utf8') as f:
            f.write('ITER,KERNEL,PRECISION,RECALL,F-SCORE,C,GAMMA\n')
            for i in vlist(range(100), 'Finding best SVM config', verbose):
                classifier = self._new_classifier()
                for kernel in ['linear', 'sigmoid', 'rbf']:
                    if kernel != 'linear':
                        for c, gamma in cs_gammas:
                            precision, recall, fscore = classifier.configure(kernel, c, gamma).fit().evaluate()
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
                            precision, recall, fscore = classifier.configure(kernel, c).fit().evaluate()
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

        classifier = SVMClassifier(max_config.x_train, max_config.y_train, max_config.x_test, max_config.y_test) \
            .configure(max_config.kernel, max_config.c, max_config.gamma).fit()
        return classifier, max_config
