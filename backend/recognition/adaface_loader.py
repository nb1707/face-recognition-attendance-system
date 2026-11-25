from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any





ADAFACE_REPO_ID = (
    "minchul/cvlface_adaface_ir101_webface4m"
)

ADAFACE_REVISION = (
    "f2b38d9e24bfe301490d8dd081d8924b102333dd"
)

BACKEND_DIR = Path(__file__).resolve().parent.parent

ADAFACE_LOCAL_DIR = (
    BACKEND_DIR
    / "model_weights"
    / "adaface"
    / "ir101_webface4m"
)


def download_adaface_repository(
    force_download: bool = False
) -> Path:
    """
    Download all source files and weights required by
    the AdaFace custom Hugging Face model.
    """
    from huggingface_hub import hf_hub_download

    ADAFACE_LOCAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files_manifest_path = hf_hub_download(
        repo_id=ADAFACE_REPO_ID,
        filename="files.txt",
        revision=ADAFACE_REVISION,
        local_dir=str(ADAFACE_LOCAL_DIR),
        force_download=force_download
    )

    with open(
        files_manifest_path,
        "r",
        encoding="utf-8"
    ) as file:
        repository_files = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    required_files = repository_files + [
        "config.json",
        "wrapper.py",
        "model.safetensors"
    ]

    required_files = list(
        dict.fromkeys(required_files)
    )

    for filename in required_files:
        print(
            f"Checking AdaFace file: {filename}"
        )

        hf_hub_download(
            repo_id=ADAFACE_REPO_ID,
            filename=filename,
            revision=ADAFACE_REVISION,
            local_dir=str(ADAFACE_LOCAL_DIR),
            force_download=force_download
        )

    return ADAFACE_LOCAL_DIR


def _remove_conflicting_models_modules() -> dict[str, Any]:
    """
    Remove existing modules named 'models' from sys.modules
    so AdaFace can import its own local models package.
    """
    removed_modules: dict[str, Any] = {}

    for module_name in list(sys.modules):
        if (
            module_name == "models"
            or module_name.startswith("models.")
        ):
            removed_modules[module_name] = (
                sys.modules.pop(module_name)
            )

    return removed_modules


def _restore_modules(
    removed_modules: dict[str, Any]
) -> None:
    for module_name, module in removed_modules.items():
        if module_name not in sys.modules:
            sys.modules[module_name] = module


def local_model_provenance():
    """Actual local checkpoint digest; never infer compatibility from dimension."""
    import hashlib
    from recognition.adaface_preprocessing import PREPROCESSING_VERSION, EMBEDDING_DIMENSION
    checkpoint = ADAFACE_LOCAL_DIR / "pretrained_model" / "model.pt"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(model_name="adaface", model_revision=ADAFACE_REVISION,
                checkpoint_sha256=digest.hexdigest(), embedding_dimension=EMBEDDING_DIMENSION,
                preprocessing_version=PREPROCESSING_VERSION)


def load_adaface_from_local():
    """Load only existing local source/weights; no HF calls, downloads or cwd edits.

    A private package namespace supports the artifact's relative imports without
    replacing an application's top-level 'models' module.
    """
    import importlib.util
    import threading
    import torch
    from omegaconf import OmegaConf

    config_path = ADAFACE_LOCAL_DIR / "pretrained_model" / "model.yaml"
    checkpoint = ADAFACE_LOCAL_DIR / "pretrained_model" / "model.pt"
    package_path = ADAFACE_LOCAL_DIR / "models" / "__init__.py"
    for path in (config_path, checkpoint, package_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local AdaFace artifact: {path}")
    config = OmegaConf.load(config_path)
    if config.color_space != "RGB" or config.output_dim != 512 or config.name != "ir101":
        raise ValueError("Local AdaFace artifact does not match RGB/IR101/512 contract")
    if config.start_from:
        raise ValueError("External AdaFace start_from checkpoints are not allowed")
    with _LOCAL_LOAD_LOCK:
        name = "_attendance_adaface_models"
        package = sys.modules.get(name)
        if package is None:
            spec = importlib.util.spec_from_file_location(name, package_path,
                    submodule_search_locations=[str(package_path.parent)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            try:
                spec.loader.exec_module(package)
            except Exception:
                sys.modules.pop(name, None)
                raise
        model = package.get_model(config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.eval()


import threading
_LOCAL_LOAD_LOCK = threading.Lock()
