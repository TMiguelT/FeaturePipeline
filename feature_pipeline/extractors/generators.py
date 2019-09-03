from feature_pipeline.core import FeatureExtractor, listener
import nltk
from nltk.corpus import words
import regex as re

from ..core import CORE
from .emitters import WordExtractor, SentenceExtractor


class TotalWordsExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.count = 0

    @listener(WordExtractor, 'word')
    def on_word(self, word):
        self.count += 1

    def process(self) -> dict:
        return {
            'word_count': self.count
        }


class TotalSentencesExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.count = 0

    @listener(SentenceExtractor, 'sentence')
    def on_word(self, word):
        self.count += 1

    def process(self) -> dict:
        return {
            'sentence_count': self.count
        }


class AvgWordPerSentenceExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.words = 0
        self.sentences = 0

    @listener(WordExtractor, 'sentence_words')
    def on_sentence_words(self, words):
        self.sentences += 1
        self.words += len(words)

    def process(self) -> dict:
        return {
            'avg_word_per_sentence': self.words / self.sentences
        }


class DictionaryFreqExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        nltk.download('words')
        self.corpus = list(words.words())
        self.in_dict = 0
        self.all = 0

    @listener(WordExtractor, 'word')
    def on_word(self, word):
        self.all += 1
        if word in self.corpus:
            self.in_dict += 1

    def process(self) -> dict:
        return {
            'in_dictionary_frequency': self.in_dict / self.all
        }


class WordExtensionExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.ext_regex = re.compile(r'\w{3,}')
        self.extended = 0
        self.all = 0

    @listener(WordExtractor, 'word')
    def on_word(self, word):
        self.all += 1
        if self.ext_regex.match(word):
            self.extended += 1

    def process(self) -> dict:
        return {
            'extended_frequency': self.extended / self.all
        }


class BosCapitalExtractor(FeatureExtractor):
    """
    Total number of beginning of sentences (BOS) characters capitalized
    """

    def __init__(self):
        super().__init__()
        self.match = None
        self.re = re.compile('/^[A-Z]+/')

    @listener(CORE, 'document')
    def on_doc(self, document):
        self.match = self.re.match(document)

    def process(self):
        return {
            'bos_capitals': self.match.end - self.match.start if self.match else 0
        }


class PunctuationCountExtractor(FeatureExtractor):
    """
    Total number of unicode punctuation characters in the document
    """

    def __init__(self):
        super().__init__()
        self.matches = []
        self.re = re.compile(r'\p{Punctuation}')

    @listener(CORE, 'document')
    def on_doc(self, document):
        self.matches = self.re.findall(document)

    def process(self):
        return {
            'punctuation_count': len(self.matches)
        }


class AllCapsFrequencyExtractor(FeatureExtractor):
    """
    Proportion of words that are in ALLCAPS
    """

    def __init__(self):
        super().__init__()
        self.words = 0
        self.matches = 0
        self.re = re.compile(r'(\p{Uppercase_Letter})+')

    @listener(WordExtractor, 'word')
    def on_word(self, word):
        self.words += 1
        match = self.re.match(word)
        if match:
            self.matches += 1

    def process(self):
        return {
            'allcaps_freq': self.matches / self.words
        }


class UpperCaseCountExtractor(FeatureExtractor):
    """
    Proportion of letters that are capitalized
    """

    def __init__(self):
        super().__init__()
        self.chars = 0
        self.matches = 0
        self.re = re.compile(r'\p{Uppercase_Letter}')

    @listener(CORE, 'character')
    def on_char(self, char):
        self.chars += 1
        match = self.re.match(char)
        if match:
            self.matches += 1

    def process(self):
        return {
            'caps_freq': self.matches / self.chars
        }


class QuotationUseExtractor(FeatureExtractor):
    """
    Binary feature: use of quotation marks
    """

    def __init__(self):
        super().__init__()
        self.quotation = False
        self.re = re.compile(r'[\p{Initial_Punctuation}\p{Final_Punctuation}]')

    @listener(CORE, 'character')
    def on_char(self, char):
        match = self.re.match(char)
        if match:
            self.quotation = True

    def process(self):
        return {
            'quotation_use': 1 if self.quotation else 0
        }
