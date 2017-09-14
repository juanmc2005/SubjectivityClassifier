from functools import reduce
from heapq import nlargest
from math import log
import postagger as tagger


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
    def _fits_pattern_1(gram):  # (ADJ, SUST, X)
        ((_, t1), (_, t2), _) = gram
        return tagger.is_adjective(t1) and tagger.is_noun(t2)

    @staticmethod
    def _fits_pattern_2(gram):  # (SUST, ADJ, X)
        ((_, t1), (_, t2), _) = gram
        return tagger.is_noun(t1) and tagger.is_adjective(t2)

    @staticmethod
    def _fits_pattern_3(gram):  # (ADV, ADJ, SUST)
        ((_, t1), (_, t2), (_, t3)) = gram
        return tagger.is_adverb(t1) and tagger.is_adjective(t2) and tagger.is_noun(t3)

    @staticmethod
    def _fits_pattern_4(gram):  # (SUST, ADV, ADJ)
        ((_, t1), (_, t2), (_, t3)) = gram
        return tagger.is_noun(t1) and tagger.is_adverb(t2) and tagger.is_adjective(t3)

    @staticmethod
    def _fits_pattern_5(gram):  # (VERB, ADV, X)
        ((_, t1), (_, t2), _) = gram
        return tagger.is_verb(t1) and tagger.is_adverb(t2)

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

    def pats(self, trigrams):
        n = len(trigrams)
        if n > 0:
            return reduce(lambda acc, gram: acc + 1 if self._fits_any_pattern(gram) else acc, trigrams, 0) / n
        else:
            return 0

    def _fits_any_pattern(self, gram):
        return self._fits_pattern_1(gram) or self._fits_pattern_2(gram) \
               or self._fits_pattern_3(gram) or self._fits_pattern_4(gram) or self._fits_pattern_5(gram)
