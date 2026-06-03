from __future__ import annotations

from ..services.step_runner import STEP_MODULES, StepRunner
from ..stores.run_store import RunStore


class RunController:
    def __init__(self, store: RunStore, settings: dict) -> None:
        self.store = store
        self.settings = settings
        self._runner: StepRunner | None = None

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

    def stop(self) -> None:
        runner = getattr(self, "_runner", None)
        if runner:
            runner.stop()
