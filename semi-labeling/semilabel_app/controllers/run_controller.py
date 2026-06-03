from __future__ import annotations

from ..domain.models import ChainStep
from ..services.pipeline_runner import PipelineRunner
from ..services.step_runner import STEP_MODULES, StepRunner
from ..stores.run_store import RunStore


class RunController:
    def __init__(self, store: RunStore, settings: dict) -> None:
        self.store = store
        self.settings = settings
        self._runner: StepRunner | None = None
        self._pipeline: PipelineRunner | None = None

    def update_settings(self, settings: dict) -> None:
        self.settings.update(settings)

    def run_step(self, step: str, flags: dict) -> None:
        if step not in STEP_MODULES:
            self.store.errorRaised.emit(f"Step not allowed: {step}")
            return
        self.store.clear_log()
        self.store.set_running(True)
        runner = StepRunner()
        runner.output.connect(self.store.append_log)
        runner.started.connect(lambda name: self.store.set_status(f"Running {name}"))
        runner.finished.connect(lambda code: self._on_step_finished(step, code))
        runner.run(step, flags)
        self._runner = runner

    def _on_step_finished(self, step: str, code: int) -> None:
        self.store.set_running(False)
        self.store.set_status(f"{step} exit {code}")

    def run_chain_05_08(self) -> None:
        db = self.settings["db_path"]
        run_id = self.settings.get("run_id", "myrun")
        steps = [
            ChainStep("step05", "steps.step05_proto.main", {"--db": db, "--run-id": run_id}),
            ChainStep("step06", "steps.step06_reliability.main", {"--db": db, "--run-id": run_id}),
            ChainStep("step07", "steps.step07_decision.main", {"--db": db, "--run-id": run_id}),
            ChainStep("step08", "steps.step08_classifier.main", {"--db": db, "--run-id": run_id, "--model-name": self.settings.get("model_name", "facebook/dinov2-giant")}),
        ]
        self.store.clear_log()
        self.store.set_running(True)
        runner = PipelineRunner()
        runner.output.connect(lambda _idx, text: self.store.append_log(text))
        runner.step_started.connect(lambda _idx, step: self.store.set_status(f"Running {step}"))
        runner.chain_finished.connect(self._on_chain_finished)
        runner.run(steps)
        self._pipeline = runner

    def _on_chain_finished(self, ok: bool) -> None:
        self.store.set_running(False)
        self.store.set_status("Chain completed" if ok else "Chain failed")

    def stop(self) -> None:
        runner = getattr(self, "_runner", None)
        pipeline = getattr(self, "_pipeline", None)
        if runner:
            runner.stop()
        if pipeline:
            pipeline.stop()
