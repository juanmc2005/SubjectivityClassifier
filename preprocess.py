"""
Base idea from: https://pybonacci.es/2015/11/24/como-hacer-analisis-de-sentimiento-en-espanol-2/
"""

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from string import punctuation

stopword_list = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')
non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str, range(10)))
