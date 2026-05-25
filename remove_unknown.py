import shutil
from pathlib import Path

from coord_to_mode import coord_to_mode

SRC = Path("click_dataset_folder")
DST = Path(".unknown")


def main():
    DST.mkdir(exist_ok=True)

    moved = 0
    for path in SRC.glob("*.png"):
        _, x_str, y_str = path.stem.split("_")
        if coord_to_mode(int(x_str), int(y_str)) != "unknown":
            continue
        shutil.move(str(path), DST / path.name)
        moved += 1
        print(f"[moved] {path.name}")

    print(f"\nmoved {moved} unknown file(s) to {DST}/")


if __name__ == "__main__":
    main()
