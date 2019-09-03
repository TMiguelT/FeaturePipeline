import pandas
import typing

from feature_pipeline.core.event import Event

# The owner for core events
CORE = object()


class ExtractionPipeline:
    def __init__(self, extractors):
        self.extractors = extractors

        # Tell the extractors about the pipeline
        for extractor in extractors:
            extractor.pipeline = self

    def emit(self, emitter, event: str, data):
        event = Event(emitter, event)
        for extractor in self.extractors:
            if event in extractor.hooks:
                for hook in extractor.hooks[event]:
                    hook(data)

    def process(self, iterable: typing.Iterable[str]) -> pandas.DataFrame:
        rows = []
        for document in iterable:
            # Trigger each processor in a cascading fashion
            self.emit(CORE, 'document', document)

            for character in document:
                self.emit(CORE, 'character', character)

            # Create a row using all extractors
            row = {}
            for extractor in self.extractors:
                row.update(extractor.process())
            rows.append(row)

        return pandas.DataFrame(rows)
