from feature_pipeline.core import FeatureExtractor, listener, CORE
from nltk.tokenize import sent_tokenize, word_tokenize

class SentenceExtractor(FeatureExtractor):
    """
    Emits "sentences" and "sentence" events
    """

    def __init__(self, language):
        super().__init__()
        self.language = language

    @listener(CORE, 'document')
    def on_doc(self, document):
        sentences = sent_tokenize(document, self.language)
        self.emit('sentences', sentences)

        for sentence in sentences:
            self.emit('sentence', sentence)


class WordExtractor(FeatureExtractor):
    """
    Emits "words" and "word" events
    """

    def __init__(self, language):
        super().__init__()
        self.language = language

    @listener(SentenceExtractor, 'sentences')
    def on_sentences(self, sentences):
        all_words = []
        for sentence in sentences:
            words = word_tokenize(sentence, self.language)
            self.emit('sentence_words', words)

            for word in words:
                all_words.append(word)
                self.emit('word', word)

        self.emit('all_words', all_words)
