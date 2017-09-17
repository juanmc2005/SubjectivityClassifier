from preprocessor import Preprocessor
from subjclassifier import SubjectivityPipeline
from itertools import product
from tqdm import tqdm


class Result:

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


classifier = SubjectivityPipeline(Preprocessor('../raw_db.txt')).preprocess()
cs = [0.01, 0.1, 1, 10, 100, 1000]
gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
cs_gammas = list(product(cs, gammas))
results = []
with open('results.txt', 'a', encoding='utf8') as f:
    f.write('ITER,KERNEL,PRECISION,RECALL,F-SCORE,C,GAMMA\n')
    for i in tqdm(range(100), 'Finding best SVM config'):
        classifier.build_classifier()
        for kernel in ['linear', 'sigmoid', 'rbf']:
            if kernel != 'linear':
                for c, gamma in cs_gammas:
                    precision, recall, fscore = classifier.fit(kernel, c, gamma).evaluate()
                    results.append(Result(
                        classifier.classifier.x_train,
                        classifier.classifier.y_train,
                        classifier.classifier.x_test,
                        classifier.classifier.y_test,
                        kernel, precision, recall, fscore, c, gamma
                    ))
                    f.write(str(i) + ',' + kernel + ',' + str(precision) + ',' + str(recall) + ',' + str(fscore) +\
                            ',' + str(c) + ',' + str(gamma) + '\n')
            else:
                for c in cs:
                    precision, recall, fscore = classifier.fit(kernel, c).evaluate()
                    results.append(Result(
                        classifier.classifier.x_train,
                        classifier.classifier.y_train,
                        classifier.classifier.x_test,
                        classifier.classifier.y_test,
                        kernel, precision, recall, fscore, c, None
                    ))
                    f.write(str(i) + ',' + kernel + ',' + str(precision) + ',' + str(recall) + ',' + str(fscore) +
                            ',' + str(c) + '\n')

max = None
for res in results:
    if max is None or (res.fscore[0] > max.fscore[0] and res.fscore[1] > max.fscore[1]):
        max = res

with open('optimal_partition.txt', 'a', encoding='utf8') as f:
    f.write('X TRAIN\n')
    for v in max.x_train:
        f.write(str(v) + '\n')
    f.write('\nY TRAIN\n')
    for v in max.y_train:
        f.write(str(v) + '\n')
    f.write('\nX TEST\n')
    for v in max.x_test:
        f.write(str(v) + '\n')
    f.write('\nY TEST\n')
    for v in max.y_test:
        f.write(str(v) + '\n')
    f.write('\n\n')
    f.write('Kernel: ' + str(max.kernel) + '\n')
    f.write('Precision: ' + str(max.precision) + '\n')
    f.write('Recall: ' + str(max.recall) + '\n')
    f.write('F-Score: ' + str(max.fscore) + '\n')
