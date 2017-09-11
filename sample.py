from subjclassifier import PreprocessorSubjClassifier
from preprocessor import Preprocessor
import numpy as np

classifier = PreprocessorSubjClassifier(Preprocessor('raw_db.txt'))
classifier.preprocess()


def print_results(i):
    print(classifier.sentences[i])
    print(classifier.processed_sentences[i])
    print(np.asmatrix(classifier.matrices[i]))
    print()

print_results(15)
print_results(800)
print_results(1500)
