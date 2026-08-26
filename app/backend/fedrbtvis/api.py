from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import (
    FastAPI,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from fedrbtvis.artifacts import ArtifactCorruptError
from fedrbtvis.config import RunConfig
from fedrbtvis.events import TrainingEvent
from fedrbtvis.legacy import LegacyRepository
from fedrbtvis.manager import (
    RunAlreadyActiveError,
    RunManager,
    RunNotFoundError,
    RunNotReadyError,
    TERMINAL_STATES,
)
from fedrbtvis.presets import PresetName, build_preset
from fedrbtvis.studies import (
    StudyConfig,
    StudyCorruptError,
    StudyNotFoundError,
    expand_study,
    prepare_study,
    read_observations,
    read_study,
    reindex_interrupted_studies,
    run_study,
)

logger = logging.getLogger(__name__)

_PRESET_NAMES: tuple[PresetName, ...] = (
    "test-fixture",
    "research-lite",
    "historical-compatible",
)
_ALLOWED_OVERRIDES = frozenset(
    {
        "seed",
        "local_epochs",
        "cycles",
        "clients_per_step",
        "checkpoint_policy",
    }
)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal[
        "test-fixture",
        "research-lite",
        "historical-compatible",
    ]
    overrides: dict[str, JsonValue] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class InvalidConfigError(ValueError):
    pass


class LegacyNotImportedError(RuntimeError):
    pass


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _event_body(event: TrainingEvent) -> dict[str, JsonValue]:
    return asdict(event)


def _validated_config(
    request: CreateRunRequest,
    data_dir: Path,
    artifact_root: Path,
) -> RunConfig:
    unexpected = sorted(set(request.overrides) - _ALLOWED_OVERRIDES)
    if unexpected:
        raise InvalidConfigError(
            f"unsupported override fields: {', '.join(unexpected)}"
        )
    base = build_preset(request.preset, data_dir, artifact_root)
    values = base.model_dump()
    values.update(request.overrides)
    try:
        return RunConfig.model_validate(values)
    except ValidationError as error:
        raise InvalidConfigError("run configuration is invalid") from error


def create_app(
    manager: RunManager,
    legacy_repository: LegacyRepository | None = None,
    legacy_error: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        reindex_interrupted_studies(application.state.study_root)
        try:
            yield
        finally:
            study_tasks = tuple(application.state.study_tasks.values())
            for task in study_tasks:
                task.cancel()
            if study_tasks:
                await asyncio.sleep(0)
            await manager.shutdown()
            if study_tasks:
                await asyncio.gather(*study_tasks, return_exceptions=True)

    app = FastAPI(title="FedRBTVis", version="1", lifespan=lifespan)
    app.state.data_dir = manager.store.root.parent / "data"
    app.state.artifact_root = manager.store.root
    app.state.study_root = manager.store.root.parent / "studies"
    app.state.study_tasks = {}
    app.state.legacy_repository = legacy_repository
    app.state.legacy_error = legacy_error

    def finish_study_task(
        completed: asyncio.Task,
        study_id: str,
    ) -> None:
        app.state.study_tasks.pop(study_id, None)
        if completed.cancelled():
            return
        try:
            completed.result()
        except Exception:
            logger.exception("study task failed before writing terminal state")

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return _error(422, "INVALID_CONFIG", "request validation failed")

    @app.exception_handler(InvalidConfigError)
    async def invalid_config_error(
        request: Request,
        error: InvalidConfigError,
    ) -> JSONResponse:
        return _error(422, "INVALID_CONFIG", str(error))

    @app.exception_handler(RunNotFoundError)
    async def run_not_found_error(
        request: Request,
        error: RunNotFoundError,
    ) -> JSONResponse:
        return _error(404, "RUN_NOT_FOUND", "run was not found")

    @app.exception_handler(RunAlreadyActiveError)
    async def run_already_active_error(
        request: Request,
        error: RunAlreadyActiveError,
    ) -> JSONResponse:
        return _error(409, "RUN_ALREADY_ACTIVE", "another run is already active")

    @app.exception_handler(RunNotReadyError)
    async def run_not_ready_error(
        request: Request,
        error: RunNotReadyError,
    ) -> JSONResponse:
        return _error(409, "RUN_NOT_READY", "run metrics are not available")

    @app.exception_handler(ArtifactCorruptError)
    async def artifact_corrupt_error(
        request: Request,
        error: ArtifactCorruptError,
    ) -> JSONResponse:
        logger.exception("corrupt run artifact", exc_info=error)
        return _error(500, "ARTIFACT_CORRUPT", "a persisted run artifact is corrupt")

    @app.exception_handler(StudyNotFoundError)
    async def study_not_found_error(
        request: Request,
        error: StudyNotFoundError,
    ) -> JSONResponse:
        return _error(404, "STUDY_NOT_FOUND", "study was not found")

    @app.exception_handler(StudyCorruptError)
    async def study_corrupt_error(
        request: Request,
        error: StudyCorruptError,
    ) -> JSONResponse:
        logger.exception("corrupt study artifact", exc_info=error)
        return _error(500, "ARTIFACT_CORRUPT", "a persisted study artifact is corrupt")

    @app.exception_handler(LegacyNotImportedError)
    async def legacy_not_imported_error(
        request: Request,
        error: LegacyNotImportedError,
    ) -> JSONResponse:
        return _error(
            404,
            "LEGACY_NOT_IMPORTED",
            "legacy observations have not been imported",
        )

    @app.get("/api/presets")
    async def presets() -> dict[str, list[dict[str, JsonValue]]]:
        items = []
        for name in _PRESET_NAMES:
            config = build_preset(
                name,
                app.state.data_dir,
                app.state.artifact_root,
            )
            items.append(
                config.model_dump(
                    mode="json",
                    exclude={"data_dir", "artifact_root"},
                )
            )
        return {"items": items}

    @app.post("/api/runs", status_code=202)
    async def create_run(request: CreateRunRequest) -> dict[str, JsonValue]:
        config = _validated_config(
            request,
            app.state.data_dir,
            app.state.artifact_root,
        )
        created = await manager.create_run(config)
        return {"run_id": created.run_id, "status": created.status}

    @app.get("/api/runs")
    async def runs() -> dict[str, list[dict[str, JsonValue]]]:
        return {"items": manager.list_runs()}

    @app.get("/api/runs/{run_id}")
    async def run(run_id: str) -> dict[str, JsonValue]:
        return manager.get_run(run_id)

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(run_id: str, response: Response) -> dict[str, JsonValue]:
        current = manager.get_run(run_id)
        if current.get("status") not in TERMINAL_STATES:
            response.status_code = 202
        return await manager.request_stop(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(
        run_id: str,
        after_sequence: int = Query(default=0, ge=0),
    ) -> dict[str, list[dict[str, JsonValue]]]:
        return {
            "items": [
                _event_body(item)
                for item in manager.events(run_id, after_sequence)
            ]
        }

    @app.get("/api/runs/{run_id}/metrics/clients")
    async def client_metrics(
        run_id: str,
    ) -> dict[str, list[dict[str, JsonValue]]]:
        return {"items": manager.client_metrics(run_id)}

    @app.get("/api/runs/{run_id}/metrics/aggregations")
    async def aggregation_metrics(
        run_id: str,
    ) -> dict[str, list[dict[str, JsonValue]]]:
        return {"items": manager.aggregation_metrics(run_id)}

    @app.websocket("/ws/runs/{run_id}")
    async def run_events(
        websocket: WebSocket,
        run_id: str,
        after_sequence: int = Query(default=0, ge=0),
    ) -> None:
        try:
            manager.get_run(run_id)
        except RunNotFoundError as error:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="RUN_NOT_FOUND",
            ) from error
        except ArtifactCorruptError as error:
            logger.exception("corrupt run artifact", exc_info=error)
            raise WebSocketException(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="ARTIFACT_CORRUPT",
            ) from error

        await websocket.accept()
        try:
            async for item in manager.subscribe(run_id, after_sequence):
                await websocket.send_json(_event_body(item))
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        except ArtifactCorruptError as error:
            logger.exception("corrupt run artifact during WebSocket replay", exc_info=error)
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="ARTIFACT_CORRUPT",
            )
        except WebSocketDisconnect:
            return

    @app.post("/api/studies", status_code=202)
    async def create_study(spec: StudyConfig) -> dict[str, JsonValue]:
        try:
            configs = expand_study(
                spec,
                app.state.data_dir,
                app.state.artifact_root,
            )
        except ValidationError as error:
            raise InvalidConfigError("study configuration is invalid") from error
        study_id = str(uuid4())
        prepare_study(study_id, configs, app.state.study_root)
        task = asyncio.create_task(
            run_study(
                study_id,
                configs,
                manager,
                app.state.study_root,
            ),
            name=f"fedrbtvis-study-{study_id}",
        )
        app.state.study_tasks[study_id] = task
        task.add_done_callback(
            lambda completed, current_id=study_id: finish_study_task(
                completed,
                current_id,
            )
        )
        return {"study_id": study_id, "status": "queued"}

    @app.get("/api/studies/{study_id}")
    async def study(study_id: str) -> dict[str, JsonValue]:
        return read_study(app.state.study_root, study_id)

    @app.get("/api/studies/{study_id}/observations")
    async def study_observations(
        study_id: str,
    ) -> dict[str, list[dict[str, JsonValue]]]:
        return {"items": read_observations(app.state.study_root, study_id)}

    @app.get("/api/observations/legacy")
    async def legacy_observations() -> dict[str, object]:
        if app.state.legacy_error is not None:
            raise ArtifactCorruptError("legacy observation repository is corrupt")
        repository = app.state.legacy_repository
        if repository is None:
            raise LegacyNotImportedError()
        return {
            "manifest": repository.manifest_summary(),
            "items": list(repository.observations()),
        }

    return app
