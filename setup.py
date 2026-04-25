from __future__ import annotations

from pathlib import Path

from setuptools import Command, find_packages, setup


ROOT = Path(__file__).resolve().parent


def _read_requirements() -> list[str]:
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return []
    requirements: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        requirements.append(item)
    return requirements


class DownloadModelsCommand(Command):
    description = "Download pretrained model assets defined in tools.download_models."
    user_options = [
        ("name=", "n", "Model preset name, comma-separated, or 'all'."),
        ("out-dir=", "o", "Destination repo root or models directory."),
        ("force", None, "Overwrite existing files."),
    ]

    def initialize_options(self) -> None:
        self.name = "all"
        self.out_dir = None
        self.force = False

    def finalize_options(self) -> None:
        self.name = str(self.name or "all")
        self.out_dir = str(self.out_dir or ROOT)
        self.force = bool(self.force)

    def run(self) -> None:
        from tools.download_models import download_named_models

        names = [item.strip() for item in self.name.split(",") if item.strip()]
        download_named_models(
            names or ["all"],
            out_dir=self.out_dir,
            force=self.force,
            log_fn=lambda msg: self.announce(msg, level=2),
        )


setup(
    name="damage-detector",
    version="0.1.0",
    description="DamageDetector model runtimes, training pipelines, and tooling.",
    # These are repo-root helper modules imported as top-level modules.
    py_modules=["torch_runtime", "device_utils"],
    packages=find_packages(
        include=[
            "inference_api",
            "inference_api.*",
            "object_detection",
            "object_detection.*",
            "segmentation",
            "segmentation.*",
            "tools",
            "tools.*",
            "ui",
            "ui.*",
        ]
    ),
    include_package_data=True,
    install_requires=_read_requirements(),
    cmdclass={"download_models": DownloadModelsCommand},
    entry_points={
        "console_scripts": [
            "damage-dino=object_detection.dino:main",
            "damage-dino-download=object_detection.dino.download:main",
            "damage-grounding-dino-image=object_detection.grounding_dino.image:main",
            "damage-grounding-dino-folder=object_detection.grounding_dino.folder:main",
            "damage-stable-dino-train=object_detection.stable_dino.train:main",
            "damage-stable-dino-infer=object_detection.stable_dino.Inference:main",
            "damage-unet=segmentation.unet.cli:main",
            "damage-unet-train=segmentation.unet.train:main",
            "damage-unet-predict=segmentation.unet.predict:main",
            "damage-sam=segmentation.sam.no_finetune.cli:main",
            "damage-sam-finetune=segmentation.sam.finetune.cli:main",
            "damage-sam-finetune-train=segmentation.sam.finetune.train:main",
            "damage-sam-finetune-test=segmentation.sam.finetune.test:main",
            "damage-sam-finetune-pseudo-label=segmentation.sam.finetune.pseudo_label:main",
            "damage-editor=ui.editor_app:main",
            "damage-cropper=ui.create_data_tools.cropper_app.main:main",
            "damage-models=tools.download_models:main",
        ]
    },
)
