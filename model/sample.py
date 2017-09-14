import numpy as np
from preprocessor import Preprocessor

from subjclassifier import PreprocessorSubjClassifier

classifier = PreprocessorSubjClassifier(Preprocessor('../sample_db.txt'))
classifier.preprocess()


def print_results(i):
    print()
    print(classifier.sentences[i])
    print(classifier.processed_sentences[i])
    print(np.asmatrix(classifier.matrices[i]))
    print(classifier.vectors[i])
    print()

print_results(0)
