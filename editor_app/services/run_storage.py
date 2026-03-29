from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from editor_app.domain.models import RunContext, RunSummary


class RunStorageService:
    def plan_run(self, *, results_root: Path, workflow: str, scope: str) -> RunContext:
        created_at = _dt.datetime.now().isoformat(timespec="seconds")
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = f"{stamp}_{workflow}_{uuid4().hex[:6]}"
        run_dir = results_root / run_id
        output_dir = run_dir / "outputs"
        data_dir = run_dir / "data"
        ctx = RunContext(
            run_id=run_id,
            workflow=str(workflow),
            scope=str(scope),
            run_dir=run_dir,
            output_dir=output_dir,
            data_dir=data_dir,
            created_at=created_at,
        )
        return ctx

    def materialize_run(
        self,
        run: RunContext,
        *,
        status: str = "queued",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        run.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run.run_id,
            "workflow": run.workflow,
            "scope": run.scope,
            "created_at": run.created_at,
            "run_dir": str(run.run_dir),
            "output_dir": str(run.output_dir),
            "status": str(status),
            "error": error,
        }
        if metadata:
            payload.update(dict(metadata))
        self._write_json(run.run_dir / "run.json", payload)

    def create_run(self, *, results_root: Path, workflow: str, scope: str) -> RunContext:
        ctx = self.plan_run(results_root=results_root, workflow=workflow, scope=scope)
        self.materialize_run(ctx)
        return ctx

    def write_request(self, run: RunContext, request_data: dict[str, Any]) -> None:
        self._write_json(run.run_dir / "request.json", request_data)

    def append_event(self, run: RunContext, event_data: dict[str, Any]) -> None:
        events_path = run.data_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_data, ensure_ascii=True, default=str) + "\n")
        message = str(event_data.get("message") or "").strip()
        if message:
            with (run.run_dir / "logs.txt").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def write_result(
        self,
        run: RunContext,
        *,
        status: str,
        payload: dict[str, Any] | None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        result_payload = {
            "run_id": run.run_id,
            "workflow": run.workflow,
            "status": str(status),
            "error": error,
            "result": dict(payload or {}),
        }
        self._write_json(run.run_dir / "result.json", result_payload)
        run_meta: dict[str, Any] = {}
        run_json_path = run.run_dir / "run.json"
        if run_json_path.is_file():
            try:
                run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))
            except Exception:
                run_meta = {}
        run_meta.update(
            {
                "run_id": run.run_id,
                "workflow": run.workflow,
                "scope": run.scope,
                "created_at": run.created_at,
                "run_dir": str(run.run_dir),
                "output_dir": str(run.output_dir),
                "status": str(status),
                "error": error,
            }
        )
        if metadata:
            run_meta.update(dict(metadata))
        self._write_json(run.run_dir / "run.json", run_meta)

        detections = []
        if payload:
            if isinstance(payload.get("display_detections"), list):
                detections = list(payload.get("display_detections") or [])
            elif isinstance(payload.get("detections"), list):
                detections = list(payload.get("detections") or [])
            elif isinstance(payload.get("results"), list):
                for item in payload.get("results") or []:
                    if isinstance(item, dict):
                        source = item.get("display_detections") if isinstance(item.get("display_detections"), list) else item.get("detections")
                        for det in source or []:
                            detections.append(dict(det))
        self._write_json(run.data_dir / "detections.json", detections)

    def list_runs(self, results_root: Path) -> list[RunSummary]:
        if not results_root.is_dir():
            return []
        runs: list[RunSummary] = []
        for run_dir in sorted(results_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            run_json = run_dir / "run.json"
            if not run_json.is_file():
                continue
            try:
                payload = json.loads(run_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            runs.append(
                RunSummary(
                    run_id=str(payload.get("run_id") or run_dir.name),
                    workflow=str(payload.get("workflow") or ""),
                    status=str(payload.get("status") or "unknown"),
                    created_at=str(payload.get("created_at") or ""),
                    run_dir=str(run_dir),
                    task_group=str(payload.get("task_group") or ""),
                    segmentation_model=str(payload.get("segmentation_model") or ""),
                    detection_model=str(payload.get("detection_model") or ""),
                    request_path=str(run_dir / "request.json") if (run_dir / "request.json").is_file() else None,
                    result_path=str(run_dir / "result.json") if (run_dir / "result.json").is_file() else None,
                    log_path=str(run_dir / "logs.txt") if (run_dir / "logs.txt").is_file() else None,
                )
            )
        return runs

    def load_run_bundle(self, run_dir: Path) -> dict[str, Any]:
        bundle: dict[str, Any] = {
            "run_dir": str(run_dir),
            "run": {},
            "request": {},
            "result": {},
            "detections": [],
        }
        for key, name in (("run", "run.json"), ("request", "request.json"), ("result", "result.json"), ("detections", "data/detections.json")):
            path = run_dir / name
            if not path.is_file():
                continue
            try:
                bundle[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                bundle[key] = {} if key != "detections" else []
        return bundle

    def list_result_items(self, run_dir: Path) -> list[dict[str, Any]]:
        bundle = self.load_run_bundle(run_dir)
        request_payload = dict(bundle.get("request") or {})
        result_payload = dict(bundle.get("result") or {})
        payload = dict(result_payload.get("result") or {})
        items: list[dict[str, Any]] = []
        request_image_path = str(request_payload.get("image_path") or "").strip()
        request_image_paths = [str(path).strip() for path in (request_payload.get("image_paths") or []) if str(path).strip()]

        def _with_defaults(entry: dict[str, Any], index: int) -> dict[str, Any]:
            item = dict(entry or {})
            if not str(item.get("image_path") or "").strip():
                if request_image_paths:
                    if index < len(request_image_paths):
                        item["image_path"] = request_image_paths[index]
                    else:
                        item["image_path"] = request_image_paths[0]
                elif request_image_path:
                    item["image_path"] = request_image_path
            item.setdefault("_item_index", index)
            item.setdefault("_run_dir", str(run_dir))
            return item

        if isinstance(payload.get("results"), list):
            for index, item in enumerate(payload.get("results") or []):
                if not isinstance(item, dict):
                    continue
                items.append(_with_defaults(dict(item), index))
            return items
        if payload:
            items.append(_with_defaults(dict(payload), 0))
            return items
        if request_image_path or request_image_paths:
            items.append(_with_defaults({}, 0))
        return items

    def list_isolate_items(self, results_root: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for run in self.list_runs(results_root):
            run_dir = Path(run.run_dir)
            for item in self.list_result_items(run_dir):
                isolate_path = str(item.get("isolate_path") or "").strip()
                if not isolate_path:
                    continue
                items.append(
                    {
                        "run_id": run.run_id,
                        "workflow": run.workflow,
                        "status": run.status,
                        "created_at": run.created_at,
                        "run_dir": run.run_dir,
                        "image_path": str(item.get("image_path") or ""),
                        "prompt": str(item.get("prompt") or ""),
                        "isolate_action": str(item.get("isolate_action") or "keep"),
                        "isolate_path": isolate_path,
                        "mask_path": str(item.get("mask_path") or ""),
                    }
                )
        return items

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str) + "\n", encoding="utf-8")
