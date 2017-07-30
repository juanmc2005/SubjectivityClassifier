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

spanish_postagger = StanfordPOSTagger(
    'res/stanford/models/spanish.tagger',
    'res/stanford/stanford-postagger.jar',
    encoding='utf8'
)


def ismodifier(label):
    return label == 'a' or label == 'r'


def tag(sentence):
    return spanish_postagger.tag(sentence.split('\n'))


def tag_sentences(sentences):
    res = []
    for sent in sentences:
        res.append(tag(sent))
    return res
