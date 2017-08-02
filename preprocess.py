"""
Base idea from: https://pybonacci.es/2015/11/24/como-hacer-analisis-de-sentimiento-en-espanol-2/
"""

import emoji
import numpy as np
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

    print(emoji.emojize(':thumbsup:', use_aliases=True))

    def format_sentences(sentences):
        print('Formatting {} sentences...'.format(len(sentences)))
        tokenized = []
        res = []
        for s in tqdm(sentences, desc='    Cleaning characters and tokenizing'):
            # Remove unwanted characters
            sent = ''.join([c for c in s if c not in non_words])
            # Tokenize sentence
            tokenized.append(word_tokenize(sent, LANG))

        print(emoji.emojize('    Tagging sentences... ' +
                            'This might take a while, feel free to take a coffee break :coffee:',
                            use_aliases=True))
        # Tag sentences
        tokenized = tagger.tag_all(tokenized)
        print(emoji.emojize('    Tagging completed. Hope you enjoyed your coffee :smile:',
                            use_aliases=True))

        for s in tqdm(tokenized, desc='    Cleaning words and stemming'):
            # Clean POS tag and add word position in sentence
            sent = [(w, t[0], i) for i, (w, t) in enumerate(s)]
            # Remove stop words
            sent = [p for p in sent if p[0] not in stopword_list]
            # Stem
            sent = [(stemmer.stem(w), t, i) for w, t, i in sent]
            if sent:
                res.append(sent)
        print('Formatting completed')
        return res

    def with_tag(label, labels, sentences):
        return [[w for (w, t, i) in sentences[i]] for i, l in enumerate(labels) if l == label]

    with open(DB_FILE, encoding='utf8') as db:
        # Split sentences into label and sentence
        sentences = [x.split('@') for x in db.readlines()]
        labels = [p[0] for p in sentences]
        sentences = format_sentences([p[1] for p in sentences])
        # Get subjective and objective sentences
        subjective = with_tag('S', labels, sentences)
        objective = with_tag('O', labels, sentences)

        # Build matrices
        calc = Calculator(subjective, objective)
        matrices = []
        for sentence in tqdm(sentences, desc='Building matrices'):
            matrix = [[], [], [], []]
            for (word, tag, pos) in sentence:
                matrix[0].append(calc.swfisf(word))
                matrix[1].append(calc.occurrences(word))
                matrix[2].append(pos)
                matrix[3].append(1 if tagger.ismodifier(tag) else 0)
                # TODO reported key words
                # TODO possible sentiment analysis
            matrices.append(matrix)

        print(emoji.emojize('Done :ok_hand:', use_aliases=True))
        print(labels[10], sentences[10])
        print(np.matrix(matrices[10]))

    return sentences, labels, matrices


preprocess()
