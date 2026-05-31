"""steps — the 9 pipeline steps, one file per step.

Each module exposes ``build_parser()`` and ``main(argv)`` and is the single
CLI entry point for that step. Run them via the orchestrator:

    python -m run_pipeline run step04 --run-id <id>

or directly:

    python -m steps.step04_core --run-id <id>
"""
