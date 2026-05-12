import shutil
from pathlib import Path

from git import Repo


DATASET_REPO = "https://github.com/rasbt/mnist-pngs"
TARGET_DIR = Path(__file__).resolve().parents[1]
CLONE_DIR = TARGET_DIR / "mnist-pngs"


def replace_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    shutil.move(str(source), str(destination))


def main() -> None:
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)

    try:
        Repo.clone_from(DATASET_REPO, CLONE_DIR)

        for name in ("train", "test", "train.csv", "test.csv"):
            replace_path(CLONE_DIR / name, TARGET_DIR / name)
    finally:
        if CLONE_DIR.exists():
            shutil.rmtree(CLONE_DIR)


if __name__ == "__main__":
    main()
