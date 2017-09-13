from functools import reduce
from math import log


class MetricsCalculator:

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
