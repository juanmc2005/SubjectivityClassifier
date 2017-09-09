from subjclassifier import PreprocessorSubjClassifier
from preprocessor import Preprocessor
import numpy as np

classifier = PreprocessorSubjClassifier(Preprocessor('raw_db.txt'))
classifier.preprocess()

print(np.asmatrix(classifier.matrices[10]))
