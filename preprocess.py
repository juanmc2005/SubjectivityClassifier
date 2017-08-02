"""
Base idea from: https://pybonacci.es/2015/11/24/como-hacer-analisis-de-sentimiento-en-espanol-2/
"""

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from string import punctuation
from tqdm import tqdm

from calculator import Calculator
import postagger as tagger


LANG = 'spanish'
DB_FILE = 'raw_db.txt'


def preprocess():

    print('Loading assets...', end=' ')

    stopword_list = stopwords.words(LANG)
    stemmer = SnowballStemmer(LANG)
    non_words = list(punctuation)
    non_words.extend(['¿', '¡'])
    non_words.extend(map(str, range(10)))

    def format_sentences(sentences):
        formatted = []
        for (tag, s) in tqdm(sentences, desc='Formatting sentences'):
            # Remove unwanted characters
            sent = ''.join([c for c in s if c not in non_words])
            # Tokenize sentence and tag words
            sent = [(w, t[0], i) for i, (w, t) in enumerate(tagger.tag(word_tokenize(sent, LANG)))]
            # Remove stop words
            sent = [p for p in sent if p[0] not in stopword_list]
            # Stem
            sent = [(stemmer.stem(w), t, i) for w, t, i in sent]
            if sent:
                formatted.append((tag, sent))
        return formatted

    def with_tag(tag, sentences):
        return [' '.join(p[1]) for p in sentences if p[0] == tag]

    with open(DB_FILE, encoding='utf8') as db:
        # Split sentences into label and sentence
        sentences = format_sentences([x.split('@') for x in db.readlines()])
        # Get subjective and objective sentences
        subjective = with_tag('S', sentences)
        objective = with_tag('O', sentences)

        print('OK')

        # Build matrices
        calc = Calculator(subjective, objective)
        matrices = []
        for (label, sentence) in tqdm(sentences, desc='Building sentence matrices'):
            matrix = [[], [], [], []]
            for (word, tag, pos) in sentence:
                matrix[0].append(calc.swfisf(word))
                matrix[1].append(calc.occurrences(word))
                matrix[2].append(pos)
                matrix[3].append(tagger.ismodifier(tag))
                # TODO reported key words
                # TODO possible sentiment analysis
            matrices.append(matrix)

        print('Done')
        print(sentences[10])


preprocess()
