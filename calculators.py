from functools import reduce
from heapq import nlargest
from math import log
from ngram_matchers import TrigramPatternMatcher, BigramPatternMatcher
from nltk import word_tokenize


class SubjectivityEstimator:

    @staticmethod
    def _occurrences(word, phrases):
        return reduce(lambda c, p: c + 1 if word in p else c, phrases, 0)

    def __init__(self, lang, non_words, subj_label, obj_label):
        self.lang = lang
        self.non_words = non_words
        self.subj_label = subj_label
        self.obj_label = obj_label
        self.db_subjective = []
        self.db_objective = []

    def with_base_sentences(self, db_subjective, db_objective):
        self.db_subjective = [self._tokenize(s) for s in db_subjective]
        self.db_objective = [self._tokenize(s) for s in db_objective]
        return self

    def subjectivity(self, word):
        s_occurrences = SubjectivityEstimator._occurrences(word, self.db_subjective)
        o_occurrences = SubjectivityEstimator._occurrences(word, self.db_objective)
        if o_occurrences == 0 and s_occurrences == 0:
            return 0
        elif o_occurrences == 0 and s_occurrences != 0:
            return 2
        else:
            return s_occurrences / o_occurrences

    def _tokenize(self, sentence):
        # Remove unwanted characters
        clean_sent = ''.join([c for c in sentence if c not in self.non_words])
        return word_tokenize(clean_sent, self.lang)

    def estimate_sentence(self, sentence):
        s_max = max([self.subjectivity(w) for w in self._tokenize(sentence)])
        if s_max > 1:
            return self.subj_label
        else:
            return self.obj_label

    def estimate_all(self, sentences):
        return [self.estimate_sentence(s) for s in sentences]


class MatrixMetricsCalculator:

    @staticmethod
    def _occurrences(word, phrases):
        return reduce(lambda c, p: c + 1 if word in p else c, phrases, 0)

    def __init__(self, subjective, objective):
        self._subjective = subjective
        self._objective = objective
        self._subjective_count = len(subjective)
        self._objective_count = len(objective)
        self._subj_word_count = 0
        for s in subjective:
            self._subj_word_count += len(s)
        self._obj_word_count = 0
        for s in objective:
            self._obj_word_count += len(s)

    def sentence_count(self):
        return self._objective_count + self._subjective_count

    def swfisf(self, word):
        fsx = self.subj_occurrences(word)
        fx = self.obj_occurrences(word) + fsx
        return (fsx / self._subjective_count) * log(self.sentence_count() / fx)

    def subj_occurrences(self, word):
        return self._occurrences(word, self._subjective)

    def frs(self, word):
        return self.subj_occurrences(word) / self._subj_word_count

    def obj_occurrences(self, word):
        return self._occurrences(word, self._objective)

    def fro(self, word):
        return self.obj_occurrences(word) / self._obj_word_count

    def occurrences(self, word):
        return self.subj_occurrences(word) + self.obj_occurrences(word)


class VectorMetricsCalculator:

    @staticmethod
    def _ngrams_rel_freq(matcher, n):
        if n > 0:
            return matcher.match_count() / n
        else:
            return 0

    @staticmethod
    def pabs(bigrams):
        return VectorMetricsCalculator._ngrams_rel_freq(BigramPatternMatcher(bigrams), len(bigrams))

    @staticmethod
    def pats(trigrams):
        return VectorMetricsCalculator._ngrams_rel_freq(TrigramPatternMatcher(trigrams), len(trigrams))

    def __init__(self, matrix):
        self._matrix = matrix
        self._height = len(matrix[0])

    def avg_max_3(self, col):
        return sum(nlargest(3, self._matrix[col])) / 3

    def rel_freq(self, col):
        return sum(self._matrix[col]) / self._height

    def col_over_col_rel_freq(self, target_col, other_col):
        target_flags = [1 for i in range(self._height) if self._matrix[target_col][i] > self._matrix[other_col][i]]
        return sum(target_flags) / self._height
