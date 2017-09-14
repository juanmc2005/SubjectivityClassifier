import postagger as tagger
from functools import reduce


class NGramPatternMatcher:

    def __init__(self, grams):
        self._grams = grams

    def matches_any(self, gram):
        raise NotImplementedError()

    def match_count(self):
        return reduce(lambda acc, gram: acc + 1 if self.matches_any(gram) else acc, self._grams, 0)


class TrigramPatternMatcher(NGramPatternMatcher):

    @staticmethod
    def matches_t1(gram):  # (ADV, ADJ, SUST)
        ((_, t1), (_, t2), (_, t3)) = gram
        return tagger.is_adverb(t1) and tagger.is_adjective(t2) and tagger.is_noun(t3)

    @staticmethod
    def matches_t2(gram):  # (SUST, ADV, ADJ)
        ((_, t1), (_, t2), (_, t3)) = gram
        return tagger.is_noun(t1) and tagger.is_adverb(t2) and tagger.is_adjective(t3)

    @staticmethod
    def matches_t3(gram):  # (VERB, ADV, ADV)
        ((_, t1), (_, t2), (_, t3)) = gram
        return tagger.is_verb(t1) and tagger.is_adverb(t2) and tagger.is_adverb(t3)

    @staticmethod
    def matches_t4(gram):  # (ADV, ADV, VERB)
        ((_, t1), (_, t2), (_, t3)) = gram
        return tagger.is_adverb(t1) and tagger.is_adverb(t2) and tagger.is_verb(t3)

    def matches_any(self, gram):
        return TrigramPatternMatcher.matches_t1(gram) or \
               TrigramPatternMatcher.matches_t2(gram) or \
               TrigramPatternMatcher.matches_t3(gram) or \
               TrigramPatternMatcher.matches_t4(gram)

    def __init__(self, grams):
        super().__init__(grams)


class BigramPatternMatcher(NGramPatternMatcher):

    @staticmethod
    def matches_b1(gram):  # (ADJ, SUST)
        ((_, t1), (_, t2)) = gram
        return tagger.is_adjective(t1) and tagger.is_noun(t2)

    @staticmethod
    def matches_b2(gram):  # (SUST, ADJ)
        ((_, t1), (_, t2)) = gram
        return tagger.is_noun(t1) and tagger.is_adjective(t2)

    @staticmethod
    def matches_b3(gram):  # (VERB, ADV)
        ((_, t1), (_, t2)) = gram
        return tagger.is_verb(t1) and tagger.is_adverb(t2)

    def matches_any(self, gram):
        return BigramPatternMatcher.matches_b1(gram) or \
               BigramPatternMatcher.matches_b2(gram) or \
               BigramPatternMatcher.matches_b3(gram)

    def __init__(self, grams):
        super().__init__(grams)
