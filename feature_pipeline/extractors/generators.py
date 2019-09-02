from feature_pipeline.core import FeatureExtractor, listener
import nltk
from nltk.corpus import words
import re

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
