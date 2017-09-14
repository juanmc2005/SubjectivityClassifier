"""
TAG REFERENCE:

a -> adjective
c -> conjunction
d -> determiner
f -> punctuation
i -> interjection
n -> noun
p -> pronoun
r -> adverb
s -> preposition
v -> verb
w -> date
z -> number

details: http://clic.ub.edu/corpus/webfm_send/18
"""

from nltk.tag.stanford import StanfordPOSTagger

_spanish_postagger = StanfordPOSTagger(
    '../res/stanford/models/spanish.tagger',
    '../res/stanford/stanford-postagger.jar',
    encoding='utf8'
)


def is_modifier(label):
    return is_adjective(label) or is_adverb(label)


def is_adjective(label):
    return label == 'a'


def is_adverb(label):
    return label == 'r'


def is_noun(label):
    return label == 'n'


def is_verb(label):
    return label == 'v'


def tag(words):
    return _spanish_postagger.tag(words)


def tag_all(sentences):
    return _spanish_postagger.tag_sents(sentences)
