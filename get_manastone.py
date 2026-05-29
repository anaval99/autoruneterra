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
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def _ocr_crop(reader, img):
    crop = img.crop(MANA_CROP_BOX)
    crop = crop.resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.LANCZOS)
    results = reader.readtext(np.array(crop), allowlist="0123456789", detail=0)
    if not results:
        return None
    digits = "".join(r for r in results if r.isdigit())
    return int(digits) if digits else None


def _resolve_path(filename):
    return filename if os.path.isabs(filename) else os.path.join("darius_dataset", filename)


def get_manastone(source="1780002947_53_95.png"):
    """source may be a filename (str/Path) or a PIL.Image."""
    if isinstance(source, Image.Image):
        img = source if source.mode == "RGB" else source.convert("RGB")
    else:
        img = Image.open(_resolve_path(str(source))).convert("RGB")
    return _ocr_crop(_get_reader(), img)


def get_manastones(inputs, batch_size=32):
    """
    Batch read mana values for a list of inputs (filenames OR PIL.Image objects).
    Returns a list of ints (or None where OCR failed), aligned with input order.

    Stacks all crops into one tall image and uses easyocr's recognize() with
    multiple regions, so the underlying recognizer runs them as a real batched
    forward pass instead of N sequential calls.
    """
    if not inputs:
        return []

    reader = _get_reader()

    crops_grey = []
    for inp in inputs:
        if isinstance(inp, Image.Image):
            img = inp if inp.mode == "RGB" else inp.convert("RGB")
        else:
            img = Image.open(_resolve_path(str(inp))).convert("RGB")
        crop = img.crop(MANA_CROP_BOX)
        crop = crop.resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.LANCZOS)
        crops_grey.append(np.array(crop.convert("L")))

    h, w = crops_grey[0].shape
    stacked = np.vstack(crops_grey)
    horizontal_list = [[0, w, i * h, (i + 1) * h] for i in range(len(inputs))]

    results = reader.recognize(
        stacked,
        horizontal_list=horizontal_list,
        free_list=[],
        allowlist="0123456789",
        detail=0,
        batch_size=batch_size,
    )

    out = []
    for r in results:
        if not r:
            out.append(None)
            continue
        digits = "".join(c for c in r if c.isdigit())
        out.append(int(digits) if digits else None)
    return out


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "1780002947_53_95.png"
    print(get_manastone(fname))
