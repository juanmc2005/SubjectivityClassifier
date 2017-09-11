import emoji
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from string import punctuation
from tqdm import tqdm

from calculator import Calculator
import postagger as tagger


class Preprocessor:

    @staticmethod
    def _sentences_with_tag(label, labels, sentences):
        return [[w for (w, t) in sentences[i]] for i, l in enumerate(labels) if l == label]

    @staticmethod
    def _verbose_list(xs, desc, verbose):
        return tqdm(xs, desc=desc) if verbose else xs

    @staticmethod
    def _verbose_print(desc, verbose):
        if verbose:
            print(desc)

    def __init__(self, filename, separator='@', subj_label='S', obj_label='O'):
        self.lang = 'spanish'
        self.filename = filename
        self.separator = separator
        self.labels = (subj_label, obj_label)
        self.stopwords = stopwords.words(self.lang)
        self.stemmer = SnowballStemmer(self.lang)
        self.non_words = list(punctuation)
        self.non_words.extend(['¿', '¡'])
        self.non_words.extend(map(str, range(10)))

    def _format_sentences(self, sentences, verbose):
        self._verbose_print('Formatting {} sentences...'.format(len(sentences)), verbose)
        tokenized = []
        res = []
        for s in self._verbose_list(sentences, '    Cleaning characters and tokenizing', verbose):
            # Remove unwanted characters
            sent = ''.join([c for c in s if c not in self.non_words])
            # Tokenize sentence
            tokenized.append(word_tokenize(sent, self.lang))

        self._verbose_print(emoji.emojize('    Tagging sentences... ' +
                                          'This might take a while, feel free to take a coffee break :coffee:',
                                          use_aliases=True), verbose)
        # Tag sentences
        tokenized = tagger.tag_all(tokenized)
        self._verbose_print(emoji.emojize('    Tagging completed. Hope you enjoyed your coffee :smile:',
                                          use_aliases=True), verbose)

        for s in self._verbose_list(tokenized, '    Cleaning words and stemming', verbose):
            # Clean POS tags
            sent = [(w, t[0]) for (w, t) in s]
            # Remove stop words
            sent = [p for p in sent if p[0] not in self.stopwords]
            # Stem
            sent = [(self.stemmer.stem(w), t) for w, t in sent]
            if sent:
                res.append(sent)
        self._verbose_print('Formatting completed', verbose)
        return res

    def preprocess(self, verbose=True):
        with open(self.filename, encoding='utf8') as db:
            # Split sentences into label and sentence
            lines = db.readlines()
            sentences = [x.split(self.separator) for x in lines]
            labels = [p[0] for p in sentences]
            sentences = self._format_sentences([p[1] for p in sentences], verbose)
            # Get subjective and objective sentences
            subjective = self._sentences_with_tag(self.labels[0], labels, sentences)
            objective = self._sentences_with_tag(self.labels[1], labels, sentences)

            # Build matrices
            calc = Calculator(subjective, objective)
            matrices = []
            for sentence in self._verbose_list(sentences, 'Building matrices', verbose):
                matrix = [[], [], [], []]
                for (word, tag) in sentence:
                    matrix[0].append(calc.swfisf(word))
                    matrix[1].append(calc.weighted_subj_occurrences(word))
                    matrix[2].append(calc.weighted_obj_occurrences(word))
                    matrix[3].append(1 if tagger.ismodifier(tag) else 0)
                matrices.append(matrix)

            self._verbose_print(emoji.emojize('Done :ok_hand:', use_aliases=True), verbose)

        return lines, sentences, labels, matrices
