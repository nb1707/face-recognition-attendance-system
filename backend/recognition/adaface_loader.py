from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoModel


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


def load_adaface_from_local():
    model_directory = (
        download_adaface_repository()
        .resolve()
    )

    internal_models_directory = (
        model_directory / "models"
    )

    if not internal_models_directory.exists():
        raise FileNotFoundError(
            "AdaFace internal models folder is missing: "
            f"{internal_models_directory}"
        )

    models_init_file = (
        internal_models_directory
        / "__init__.py"
    )

    if not models_init_file.exists():
        raise FileNotFoundError(
            "AdaFace models/__init__.py is missing."
        )

    previous_working_directory = Path.cwd()
    model_directory_string = str(model_directory)

    removed_modules = (
        _remove_conflicting_models_modules()
    )

    path_added = False

    try:
        # Put AdaFace repository first in Python import path.
        if (
            not sys.path
            or sys.path[0] != model_directory_string
        ):
            sys.path.insert(
                0,
                model_directory_string
            )
            path_added = True

        os.chdir(model_directory)

        # Validate that Python resolves AdaFace's own package.
        import models

        models_file = getattr(
            models,
            "__file__",
            None
        )

        print(
            "AdaFace models package:",
            models_file
        )

        if not hasattr(models, "get_model"):
            raise ImportError(
                "Python loaded the wrong 'models' package. "
                f"Resolved package: {models_file}"
            )

        model = AutoModel.from_pretrained(
            str(model_directory),
            trust_remote_code=True,
            local_files_only=True
        )

        return model

    finally:
        os.chdir(
            previous_working_directory
        )

        if (
            path_added
            and sys.path
            and sys.path[0]
            == model_directory_string
        ):
            sys.path.pop(0)

        # Remove AdaFace temporary top-level package.
        for module_name in list(sys.modules):
            if (
                module_name == "models"
                or module_name.startswith("models.")
            ):
                sys.modules.pop(
                    module_name,
                    None
                )

        _restore_modules(
            removed_modules
        )