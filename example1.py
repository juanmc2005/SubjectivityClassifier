from subjpipeline import Pipeline
from preprocessor import Preprocessor

"""

Este ejemplo muestra como usar el pipeline para preprocesar una base de datos dada y encontrar el modelo SVM optimo

"""


pipeline = Pipeline(Preprocessor('bd/bd.txt'))\
    .preprocess()\
    .optimal_svm()\
    .dump()
