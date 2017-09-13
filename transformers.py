from heapq import nlargest
from functools import reduce
import postagger as tagger

COL_SWFISF = 0
COL_FRS = 1
COL_FRO = 2
COL_MODIFIER = 3


def sentence_to_matrix(sentence, calculator, tagger):
    matrix = [[], [], [], []]
    for (word, tag) in sentence:
        matrix[COL_SWFISF].append(calculator.swfisf(word))
        matrix[COL_FRS].append(calculator.frs(word))
        matrix[COL_FRO].append(calculator.fro(word))
        matrix[COL_MODIFIER].append(1 if tagger.is_modifier(tag) else 0)
    return matrix


def matrix_to_vector(matrix, trigrams):
    height = len(matrix[COL_FRS])
    return [
        sum(nlargest(3, matrix[COL_SWFISF])) / 3,
        sum(matrix[COL_FRS]) / height,
        sum(matrix[COL_FRO]) / height,
        _frs_over_fro_rate(matrix, height),  # TODO NOT WORKING
        sum(matrix[COL_MODIFIER]) / height,
        _pats(trigrams)  # TODO NOT WORKING
    ]


def _pats(trigrams):
    n = sum([1 for _ in trigrams])
    if n > 0:
        return reduce(lambda acc, gram: acc + 1 if _fits_any_pattern(gram) else acc, trigrams, 0) / n
    else:
        return 0


def _fits_any_pattern(gram):
    return _fits_pattern_1(gram) or _fits_pattern_2(gram) \
           or _fits_pattern_3(gram) or _fits_pattern_4(gram) or _fits_pattern_5(gram)


def _fits_pattern_1(gram):  # (ADJ, SUST, X)
    ((_, t1), (_, t2), _) = gram
    return tagger.is_adjective(t1) and tagger.is_noun(t2)


def _fits_pattern_2(gram):  # (SUST, ADJ, X)
    ((_, t1), (_, t2), _) = gram
    return tagger.is_noun(t1) and tagger.is_adjective(t2)


def _fits_pattern_3(gram):  # (ADV, ADJ, SUST)
    ((_, t1), (_, t2), (_, t3)) = gram
    return tagger.is_adverb(t1) and tagger.is_adjective(t2) and tagger.is_noun(t3)


def _fits_pattern_4(gram):  # (SUST, ADV, ADJ)
    ((_, t1), (_, t2), (_, t3)) = gram
    return tagger.is_noun(t1) and tagger.is_adverb(t2) and tagger.is_adjective(t3)


def _fits_pattern_5(gram):  # (VERB, ADV, X)
    ((_, t1), (_, t2), _) = gram
    return tagger.is_verb(t1) and tagger.is_adverb(t2)


def _frs_over_fro_rate(matrix, height):
    sum([1 for i in range(height) if matrix[COL_FRS][i] > matrix[COL_FRO][i]]) / height
