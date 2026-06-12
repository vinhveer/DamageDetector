"""steps — the pipeline steps, one file per step.

Each module exposes ``build_parser()`` and ``main(argv)`` and is the single
CLI entry point for that step. The supported client path drives them via:

    python -m client_pipeline recommended --input-dir <imgs> --output-dir <out> --run-id <id>

or run a step directly:

    python -m steps.step04_core --run-id <id>
"""
