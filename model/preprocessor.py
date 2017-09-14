from string import punctuation

import emoji
import postagger as tagger
from calculators import MatrixMetricsCalculator
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from tqdm import tqdm
from transformers import sentence_to_matrix, matrix_to_vector


class Preprocessor:

    @staticmethod
    def _sentences_with_tag(label, labels, sentences):
        return [[w for (w, t) in sentences[i]] for i, l in enumerate(labels) if l == label]

    @staticmethod
    def _verbose_list(xs, desc, verbose):
        return tqdm(xs, desc=desc) if verbose else xs

    @staticmethod
    def _verbose_print(desc, verbose):
        if verbose:
            print(desc)

    def __init__(self, filename, separator='@', subj_label='S', obj_label='O'):
        self.lang = 'spanish'
        self.filename = filename
        self.separator = separator
        self.labels = (subj_label, obj_label)
        self.stopwords = stopwords.words(self.lang)
        self.stemmer = SnowballStemmer(self.lang)
        self.non_words = list(punctuation)
        self.non_words.extend(['¿', '¡'])
        self.non_words.extend(map(str, range(10)))

    def _format_sentences(self, sentences, verbose):
        self._verbose_print('Formatting {} sentences...'.format(len(sentences)), verbose)
        tokenized = []
        res = []
        for s in self._verbose_list(sentences, '    Cleaning and tokenizing', verbose):
            # Remove unwanted characters
            sent = ''.join([c for c in s if c not in self.non_words])
            # Tokenize sentence
            tokenized.append(word_tokenize(sent, self.lang))

        self._verbose_print(emoji.emojize('    Tagging sentences... ' +
                                          'This might take a while, feel free to take a coffee break :coffee:',
                                          use_aliases=True), verbose)
        # Tag sentences
        tokenized = tagger.tag_all(tokenized)
        self._verbose_print(emoji.emojize('    Tagging completed. Hope you enjoyed your coffee :smile:',
                                          use_aliases=True), verbose)

        bigrams = []
        trigrams = []
        for s in self._verbose_list(tokenized, '    Extracting n-grams and stemming', verbose):
            # Clean POS tags
            sent = [(w, t[0]) for (w, t) in s]
            # Extract bigrams and trigrams
            bigrams.append(list(ngrams(sent, 2)))
            trigrams.append(list(ngrams(sent, 3)))
            # Remove stop words
            sent = [p for p in sent if p[0] not in self.stopwords]
            # Stem
            sent = [(self.stemmer.stem(w), t) for w, t in sent]
            if sent:
                res.append(sent)

        self._verbose_print('Formatting completed', verbose)
        return res, bigrams, trigrams

    def assemble_matrices(self, sentences, calculator, verbose):
        matrices = []
        for sentence in self._verbose_list(sentences, 'Building matrices', verbose):
            matrices.append(sentence_to_matrix(sentence, calculator))
        return matrices

    def assemble_vectors(self, matrices, bigrams, trigrams, verbose):
        vectors = []
        for i, matrix in self._verbose_list(enumerate(matrices), 'Building vectors', verbose):
            vectors.append(matrix_to_vector(matrix, bigrams[i], trigrams[i]))
        return vectors

    def preprocess(self, verbose=True):
        with open(self.filename, encoding='utf8') as db:
            # Split sentences into label and sentence
            lines = db.readlines()
            sentences = [x.split(self.separator) for x in lines]
            labels = [p[0] for p in sentences]
            sentences, bigrams, trigrams = self._format_sentences([p[1] for p in sentences], verbose)
            # Get subjective and objective sentences
            subjective = self._sentences_with_tag(self.labels[0], labels, sentences)
            objective = self._sentences_with_tag(self.labels[1], labels, sentences)
            metrics = MatrixMetricsCalculator(subjective, objective)
            matrices = self.assemble_matrices(sentences, metrics, verbose)
            vectors = self.assemble_vectors(matrices, bigrams, trigrams, verbose)
            self._verbose_print(emoji.emojize('Done :ok_hand:', use_aliases=True), verbose)

        return lines, sentences, labels, matrices, vectors
