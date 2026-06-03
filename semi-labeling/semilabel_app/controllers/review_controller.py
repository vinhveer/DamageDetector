from __future__ import annotations

from PySide6 import QtCore

from ..config.defaults import LABELS
from ..services import db_service
from ..services.handoff_service import read_handoff_json, write_handoff_json
from ..services.write_service import commit_corrections, commit_session
from ..stores.review_store import ReviewStore


class ReviewController:
    def __init__(self, store: ReviewStore, settings: dict) -> None:
        self.store = store
        self.settings = settings
        self._workers: list[object] = []

    def update_settings(self, settings: dict) -> None:
        self.settings.update(settings)

    def _db_worker(self, fn, *args, on_done=None, **kwargs) -> None:
        if hasattr(db_service, "DbWorker") and QtCore.QCoreApplication.instance() is not None:
            worker = db_service.DbWorker(fn, *args, **kwargs)
            worker.signals.finished.connect(on_done or (lambda _result: None))
            worker.signals.error.connect(self.store.errorRaised)
            worker.signals.finished.connect(lambda _result, w=worker: self._release_worker(w))
            worker.signals.error.connect(lambda _message, w=worker: self._release_worker(w))
            self._workers.append(worker)
            QtCore.QThreadPool.globalInstance().start(worker)
            return
        try:
            result = fn(*args, **kwargs)
            if on_done:
                on_done(result)
        except Exception as exc:
            self.store.errorRaised.emit(str(exc))

    def _release_worker(self, worker: object) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def load_queue(self, queue_type: str = "", sample_percent: int | float | None = None) -> None:
        sample = float(sample_percent if sample_percent is not None else self.settings.get("sample_percent", 0)) / 100.0
        self.store.mode = "queue"
        self._db_worker(
            db_service.list_queue,
            self.settings["db_path"],
            self.settings.get("run_id", "myrun"),
            self.settings.get("image_root", ""),
            queue_type,
            sample,
            on_done=lambda payload: self.store.set_queue(payload["items"]),
        )

    def load_cleaned(
        self,
        final_label: str = "",
        decision_type: str = "",
        limit: int | None = None,
        offset: int = 0,
    ) -> None:
        self.store.mode = "cleaned"
        page_limit = int(limit if limit is not None else self.settings.get("cleaned_limit", 500))
        self._db_worker(
            db_service.list_cleaned,
            self.settings["db_path"],
            self.settings.get("run_id", "myrun"),
            self.settings.get("image_root", ""),
            final_label,
            decision_type,
            page_limit,
            int(offset or 0),
            on_done=lambda payload: self.store.set_cleaned(
                payload["items"],
                total=payload.get("total", 0),
                filtered_total=payload.get("filtered_total", payload.get("filtered", 0)),
                offset=payload.get("offset", 0),
                limit=payload.get("limit", page_limit),
            ),
        )

    def fetch_image_boxes(self, image_rel_path: str, on_done) -> None:
        """Async-load every tight box on one source image (for the context overlay)."""
        self._db_worker(
            db_service.list_image_boxes,
            self.settings["db_path"],
            self.settings.get("run_id", "myrun"),
            image_rel_path,
            on_done=on_done,
        )

    def decide_current(self, label: str) -> None:
        item = self.store.current_item()
        if item is None:
            return
        result_id = int(getattr(item, "result_id"))
        suggested = str(getattr(item, "suggested_label", getattr(item, "final_label", "")) or "")
        action = "manual_reject" if label == "reject" else ("manual_accept" if label == suggested else "manual_relabel")
        self.store.set_decision(
            result_id,
            {
                "resultId": result_id,
                "action": action,
                "previousLabel": suggested,
                "newLabel": label,
            },
        )

    def accept_suggestion(self) -> None:
        item = self.store.current_item()
        if item is not None:
            self.decide_current(str(getattr(item, "suggested_label", "") or "reject"))

    def next_item(self) -> None:
        self.store.set_index(self.store.current_index + 1)

    def prev_item(self) -> None:
        self.store.set_index(self.store.current_index - 1)

    def labels(self) -> list[str]:
        return list(self.settings.get("labels") or LABELS)

    def commit_pending_decisions(self, reviewer: str = "", notes: str = "") -> dict:
        run_id = self.settings.get("run_id", "myrun")
        request = {
            "type": "review_request",
            "db": self.settings["db_path"],
            "run_id": run_id,
            "reviewer": reviewer,
            "notes": notes,
            "decisions": list(self.store.pending_decisions.values()),
        }
        request_path = write_handoff_json(self.settings["db_path"], request, kind="review", run_id=run_id)
        applied = read_handoff_json(request_path)
        payload = commit_session(
            applied["db"],
            applied["run_id"],
            list(applied.get("decisions") or []),
            reviewer=str(applied.get("reviewer") or ""),
            notes=str(applied.get("notes") or ""),
        )
        payload["handoffJson"] = str(request_path)
        if payload.get("committed"):
            self.store.clear_pending_decisions()
        return payload

    def update_cleaned(self, item: object, label: str) -> dict:
        result_id = int(getattr(item, "result_id"))
        previous = str(getattr(item, "final_label", "") or "")
        final_label = str(label or "").strip().lower()
        if not final_label:
            return {"error": "Invalid newLabel"}
        self.store.set_correction(
            result_id,
            {
                "resultId": result_id,
                "action": "manual_reject" if final_label == "reject" else "manual_relabel",
                "previousLabel": previous,
                "newLabel": final_label,
            },
        )
        return {"queued": True, "resultId": result_id, "finalLabel": final_label}

    def commit_pending_corrections(self, reviewer: str = "", notes: str = "") -> dict:
        run_id = self.settings.get("run_id", "myrun")
        request = {
            "type": "review_request",
            "db": self.settings["db_path"],
            "run_id": run_id,
            "reviewer": reviewer,
            "notes": notes,
            "corrections": list(self.store.pending_corrections.values()),
        }
        request_path = write_handoff_json(self.settings["db_path"], request, kind="review", run_id=run_id)
        applied = read_handoff_json(request_path)
        payload = commit_corrections(
            applied["db"],
            applied["run_id"],
            list(applied.get("corrections") or []),
            reviewer=str(applied.get("reviewer") or ""),
            notes=str(applied.get("notes") or ""),
        )
        payload["handoffJson"] = str(request_path)
        if payload.get("committed"):
            self.store.clear_pending_corrections()
        return payload
