import numpy as np
from preprocessor import Preprocessor

from subjclassifier import PreprocessorSubjClassifier

classifier = PreprocessorSubjClassifier(Preprocessor('raw_db.txt'))
classifier.preprocess()


def print_results(i):
    print(classifier.sentences[i])
    print(classifier.processed_sentences[i])
    print(np.asmatrix(classifier.matrices[i]))
    print(classifier.vectors[i])
    print()

print_results(800)
print_results(900)
print_results(1000)
print_results(1100)
