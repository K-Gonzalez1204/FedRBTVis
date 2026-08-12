from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypeAlias
from uuid import uuid4

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

EventType = Literal[
    "run.started",
    "client.started",
    "client.completed",
    "aggregation.completed",
    "run.stop_requested",
    "run.stopped",
    "run.completed",
    "run.failed",
]


@dataclass(frozen=True)
class TrainingEvent:
    schema_version: int
    event_id: str
    run_id: str
    sequence: int
    type: EventType
    created_at: str
    payload: dict[str, JsonValue]


class EventEmitter:
    def __init__(
        self,
        run_id: str,
        sink: Callable[[TrainingEvent], None],
    ) -> None:
        self._run_id = run_id
        self._sink = sink
        self._sequence = 0

    def emit(
        self,
        event_type: EventType,
        payload: Mapping[str, JsonValue],
    ) -> TrainingEvent:
        self._sequence += 1
        event = TrainingEvent(
            schema_version=1,
            event_id=str(uuid4()),
            run_id=self._run_id,
            sequence=self._sequence,
            type=event_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            payload=dict(payload),
        )
        self._sink(event)
        return event
