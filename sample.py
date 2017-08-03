from subjclassifier import PreprocessorSubjClassifier
from preprocessor import Preprocessor

classifier = PreprocessorSubjClassifier(Preprocessor('raw_db.txt'))
classifier.preprocess()
