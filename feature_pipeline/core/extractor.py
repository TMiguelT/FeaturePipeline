from abc import ABC, abstractmethod
import inspect
from collections import defaultdict
import typing

from .pipeline import ExtractionPipeline
from .event import Event


def listener(class_, event):
    """
    Decorator that can be used by extractors to inform the pipeline which actions they are listening for
    """
    event = Event(owner=class_, event=event)

    def wrapper(func):
        func.fp_listens = event
        return func

    return wrapper


class FeatureExtractor(ABC):
    """
    Base for extractors
    """

    def __init__(self):
        # We need an association with the parent pipeline in order to emit events
        self.pipeline: typing.Optional[ExtractionPipeline] = None

        # Hooks is a dictionary mapping events to a list of listeners for that event
        self.hooks = defaultdict(list)

        # Find all hooks this extractor defines
        for name, fn in inspect.getmembers(self):
            if hasattr(fn, 'fp_listens'):
                self.hooks[fn.fp_listens].append(fn)

    def emit(self, event: str, data):
        """
        Emits an event, allowing downstream extractors to use this data
        :param event: The event name to broadcast
        :param data: Any data to emit to listeners
        """
        self.pipeline.emit(self.__class__, event, data)

    def process(self) -> dict:
        """
        This method returns a dictionary of values that will be used in a feature vector
        """
        return {}
