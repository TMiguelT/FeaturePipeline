from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Event:
    owner: type
    event: str
