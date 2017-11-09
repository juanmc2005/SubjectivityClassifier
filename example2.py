from subjpipeline import Pipeline
from preprocessor import Preprocessor

"""

Este ejemplo utiliza el pipeline para cargar el SVM entrenado y mostrar su evaluación individual

"""


pipeline = Pipeline(Preprocessor('bd/bd.txt')).load_svm()
scores = pipeline.classifier.evaluate()

print("""
                        Objetiva                        Subjetiva

    Precision           {0:.5f}                         {1:.5f}
    Recall              {2:.5f}                         {3:.5f}
    F-Score             {4:.5f}                         {5:.5f}
""".format(scores[0][0], scores[0][1], scores[1][0], scores[1][1], scores[2][0], scores[2][1]))
