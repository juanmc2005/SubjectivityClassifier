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

import re
from nltk.tag.stanford import StanfordPOSTagger

spanish_postagger = StanfordPOSTagger(
    'res/stanford/models/spanish.tagger',
    'res/stanford/stanford-postagger.jar',
    encoding='utf8'
)


def tag_sentences(sentences):
    for sent in sentences:

        words = re.sub('[,.?/@#]', '', sent).split()
        tagged_words = spanish_postagger.tag(words)

        for (word, tag) in tagged_words:
            print(word + ' ' + tag)

        print('')
