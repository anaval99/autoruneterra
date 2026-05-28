import shutil
from pathlib import Path

from coord_to_mode import coord_to_mode
from datacollector import load_config


def main():
    config = load_config()
    src = Path(config["output_folder"])
    dst = Path(".unknown")
    dst.mkdir(exist_ok=True)

    moved = 0
    for path in src.glob("*.png"):
        _, x_str, y_str = path.stem.split("_")
        if coord_to_mode(int(x_str), int(y_str)) != "unknown":
            continue
        shutil.move(str(path), dst / path.name)
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            shutil.move(str(sidecar), dst / sidecar.name)
        moved += 1
        print(f"[moved] {path.name}")

    print(f"\nmoved {moved} unknown sample(s) from {src}/ to {dst}/")


if __name__ == "__main__":
    main()
