import numpy as np
from preprocessor import Preprocessor
from subjclassifier import SubjectivityPipeline
from itertools import product

classifier = SubjectivityPipeline(Preprocessor('../raw_db.txt')).preprocess()
# STAGE 1
for kernel in ['linear', 'sigmoid', 'rbf', 'poly']:
    print()
    print(kernel)
    classifier.fit(kernel).evaluate()
''' STAGE 2
cs = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
for c, gamma in product(cs, gammas):
    print()
    print('C = ' + str(c) + '    gamma = ' + str(gamma))
    classifier.fit(c, gamma).evaluate()
'''
''' STAGE 3
for i in range(10):
    print()
    print()
    print('Iteration ' + str(i))
    print('RBF')
    classifier.fit('rbf', 10, 0.1).evaluate()
    print()
    print('Sigmoid')
    classifier.fit('sigmoid', 1000, 0.0001).evaluate()
'''


def print_results(i):
    print()
    print(classifier.sentences[i])
    print(classifier.processed_sentences[i])
    print(np.asmatrix(classifier.matrices[i]))
    print(classifier.vectors[i])
    print()
