from feature_pipeline.core import ExtractionPipeline
from feature_pipeline.extractors import generators, emitters


def test_two_docs():
    docs = [
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua',
        'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat'
    ]

    pipeline = ExtractionPipeline([
        emitters.WordExtractor('english'),
        emitters.SentenceExtractor('english'),

        generators.AvgWordPerSentenceExtractor(),
        generators.DictionaryFreqExtractor(),
        generators.TotalSentencesExtractor(),
        generators.TotalWordsExtractor(),
        generators.WordExtensionExtractor(),
        generators.BosCapitalExtractor(),
        generators.PunctuationCountExtractor(),
        generators.AllCapsFrequencyExtractor(),
        generators.UpperCaseCountExtractor(),
        generators.QuotationUseExtractor()
    ])

    df = pipeline.process(docs)

    # There should be a column for each generator (in this case)
    assert len(df.columns) == 10

    # There should be 2 rows
    assert len(df) == 2
