from calculators import VectorMetricsCalculator
import postagger as tagger


COL_SWFISF = 0
COL_FRS = 1
COL_FRO = 2
COL_MODIFIER = 3


def sentence_to_matrix(sentence, calculator):
    matrix = [[], [], [], []]
    for (word, tag) in sentence:
        matrix[COL_SWFISF].append(calculator.swfisf(word))
        matrix[COL_FRS].append(calculator.frs(word))
        matrix[COL_FRO].append(calculator.fro(word))
        matrix[COL_MODIFIER].append(1 if tagger.is_modifier(tag) else 0)
    return matrix


def matrix_to_vector(matrix, trigrams):
    calculator = VectorMetricsCalculator(matrix)
    return [
        calculator.avg_max_3(COL_SWFISF),
        calculator.rel_freq(COL_FRS),
        calculator.rel_freq(COL_FRO),
        calculator.col_over_col_rel_freq(COL_FRS, COL_FRO),
        calculator.rel_freq(COL_MODIFIER),
        calculator.pats(trigrams)
    ]
