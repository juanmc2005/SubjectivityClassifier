from functools import reduce
from heapq import nlargest
from math import log
from ngram_matchers import TrigramPatternMatcher, BigramPatternMatcher


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
