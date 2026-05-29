import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
from PIL import Image


# region of the 200x500 dataset crop containing the player's available mana digit
# tweak if OCR misses — slightly larger is fine, smaller risks clipping the glyph
MANA_CROP_BOX = (5, 340, 70, 410)  # (left, top, right, bottom)
UPSCALE = 4

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def get_manastone(filename="1780002947_53_95.png"):
    path = filename if os.path.isabs(filename) else os.path.join("darius_dataset", filename)
    img = Image.open(path).convert("RGB")

    crop = img.crop(MANA_CROP_BOX)
    crop = crop.resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.LANCZOS)

    reader = _get_reader()
    results = reader.readtext(np.array(crop), allowlist="0123456789", detail=0)

    if not results:
        return None
    digits = "".join(r for r in results if r.isdigit())
    return int(digits) if digits else None


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "1780002947_53_95.png"
    print(get_manastone(fname))
