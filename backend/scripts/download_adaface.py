import os
import sys


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(
        0,
        BACKEND_DIR
    )


from recognition.adaface_loader import (
    download_adaface_repository
)


def main():
    model_directory = (
        download_adaface_repository()
    )

    print()
    print("AdaFace repository downloaded.")
    print("Location:", model_directory)


if __name__ == "__main__":
    main()