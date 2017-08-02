from functools import reduce
from math import log


class Calculator:

    @staticmethod
    def _occurrences(word, phrases):
        return reduce(lambda c, p: c + 1 if word in p else c, phrases, 0)

    def __init__(self, subjective, objective):
        self._subjective = subjective
        self._objective = objective
        self._subjective_count = len(subjective)
        self._objective_count = len(objective)

    def sentence_count(self):
        return self._objective_count + self._subjective_count

    def swfisf(self, word):
        fsx = self._occurrences(word, self._subjective)
        # TODO fx can be 0!!!!!!!
        fx = self._occurrences(word, self._objective) + fsx
        return (fsx / self._subjective_count) * log(self.sentence_count() / fx)

    def occurrences(self, word):
        return self._occurrences(word, self._subjective) + self._occurrences(word, self._objective)
