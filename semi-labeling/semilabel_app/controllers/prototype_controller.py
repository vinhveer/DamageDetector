from __future__ import annotations

from PySide6 import QtCore

from ..domain.models import ChainStep
from ..services import db_service
from ..services.pipeline_runner import PipelineRunner
from ..services.step_runner import StepRunner
from ..stores.prototype_store import PrototypeStore
from ..stores.run_store import RunStore


class PrototypeController:
    def __init__(self, store: PrototypeStore, run_store: RunStore, settings: dict) -> None:
        self.store = store
        self.run_store = run_store
        self.settings = settings
        self._workers: list[object] = []
        self._runner: StepRunner | None = None
        self._pipeline: PipelineRunner | None = None

    def update_settings(self, settings: dict) -> None:
        self.settings.update(settings)

    def _db_worker(self, fn, *args, on_done=None, **kwargs) -> None:
        worker = db_service.DbWorker(fn, *args, **kwargs)
        worker.signals.finished.connect(on_done or (lambda _result: None))
        worker.signals.error.connect(self.store.errorRaised)
        worker.signals.finished.connect(lambda _result, w=worker: self._release_worker(w))
        worker.signals.error.connect(lambda _message, w=worker: self._release_worker(w))
        self._workers.append(worker)
        QtCore.QThreadPool.globalInstance().start(worker)

    def _release_worker(self, worker: object) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def refresh(self) -> None:
        self.store.statusChanged.emit("Loading prototype candidates")

        def done(payload: dict) -> None:
            self._db_worker(
                db_service.latest_prototype,
                self.settings["db_path"],
                self.settings.get("run_id", "myrun"),
                on_done=lambda latest: self.store.set_candidates(payload["items"], latest.get("prototype")),
            )

        self._db_worker(
            db_service.list_prototype_candidates,
            self.settings["db_path"],
            self.settings.get("run_id", "myrun"),
            self.settings.get("image_root", ""),
            float(self.settings.get("reject_below", 0.5)),
            int(self.settings.get("per_band", 200)),
            on_done=done,
        )

    def build_step05_flags(self) -> dict:
        prototypes = []
        rejects = []
        for result_id, pick in self.store.picks.items():
            label = str(pick.get("label") or "reject")
            target = rejects if label == "reject" or pick.get("is_reject") else prototypes
            target.append(f"{int(result_id)}:{label}")
        return {
            "--db": self.settings["db_path"],
            "--run-id": self.settings.get("run_id", "myrun"),
            "--model-name": self.settings.get("model_name", "facebook/dinov2-giant"),
            "--view-name": self.settings.get("view_name", "tight"),
            "--prototype": ",".join(prototypes),
            "--reject": ",".join(rejects),
        }

    def run_step05_only(self) -> None:
        self.run_store.clear_log()
        self.run_store.set_running(True)
        runner = StepRunner()
        runner.output.connect(self.run_store.append_log)
        runner.finished.connect(lambda code: self._on_single_step_finished("step05", code))
        runner.run("step05", self.build_step05_flags())
        self._runner = runner

    def _chain_steps(self) -> list[ChainStep]:
        db = self.settings["db_path"]
        run_id = self.settings.get("run_id", "myrun")
        model_name = self.settings.get("model_name", "facebook/dinov2-giant")
        return [
            ChainStep("step05", "steps.step05_proto.main", self.build_step05_flags()),
            ChainStep("step06", "steps.step06_reliability.main", {"--db": db, "--run-id": run_id}),
            ChainStep("step07", "steps.step07_decision.main", {"--db": db, "--run-id": run_id}),
            ChainStep(
                "step08",
                "steps.step08_classifier.main",
                {"--db": db, "--run-id": run_id, "--model-name": model_name},
            ),
        ]

    def run_prototype_chain(self) -> None:
        self.run_store.clear_log()
        self.run_store.set_running(True)
        runner = PipelineRunner()
        runner.output.connect(lambda _idx, text: self.run_store.append_log(text))
        runner.step_started.connect(self._on_chain_step_started)
        runner.step_finished.connect(lambda _idx, code: self.run_store.append_log(f"[step exit {code}]\n"))
        runner.chain_finished.connect(self._on_chain_finished)
        runner.run(self._chain_steps())
        self._pipeline = runner

    def _on_single_step_finished(self, step: str, code: int) -> None:
        self.run_store.set_running(False)
        self.run_store.set_status(f"{step} exit {code}")

    def _on_chain_step_started(self, _index: int, step: str) -> None:
        self.run_store.set_step(step)
        self.run_store.set_status(f"Running {step}")

    def _on_chain_finished(self, ok: bool) -> None:
        self.run_store.set_running(False)
        self.run_store.set_status("Chain completed" if ok else "Chain failed")
