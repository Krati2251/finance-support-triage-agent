"""
OCR module — extract text from uploaded images.

Uses EasyOCR for robust multilingual text recognition.
No external system dependencies required (unlike pytesseract/Tesseract).
"""

import io
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# -------------------- Lazy-loaded EasyOCR reader --------------------
# EasyOCR downloads model weights on first use (~100 MB).
# We initialise it lazily so the app starts fast and the model
# is only loaded when OCR is actually needed.

_reader = None
_easyocr = None

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp",
}


def _get_reader():
    """Return a cached EasyOCR reader instance (English)."""
    global _reader, _easyocr
    if _reader is None:
        logger.info("🔤 Initialising EasyOCR reader (first call — may download models)...")
        import easyocr as _easyocr_module
        _easyocr = _easyocr_module
        _reader = _easyocr.Reader(["en"], gpu=False)
        logger.info("✅ EasyOCR reader ready.")
    return _reader


def extract_text_from_image(image_file: bytes | io.BytesIO) -> str:
    """
    Extract text from an image file using EasyOCR.

    Args:
        image_file: Raw image bytes or a BytesIO buffer (e.g. from an upload).

    Returns:
        A single string with all recognised text, paragraphs separated
        by newlines.

    Raises:
        ValueError: If the image cannot be opened or contains no text.
    """
    # ---- Load image ----
    try:
        if isinstance(image_file, bytes):
            image_file = io.BytesIO(image_file)

        image = Image.open(image_file)
        image = image.convert("RGB")  # ensure 3-channel
        image_np = np.array(image)
    except Exception as e:
        raise ValueError(f"Could not open the image file: {e}")

    # ---- Run OCR ----
    reader = _get_reader()
    results = reader.readtext(image_np, detail=0, paragraph=True)

    if not results:
        raise ValueError(
            "No text could be extracted from the image. "
            "Please upload a clearer image or type the email text manually."
        )

    extracted_text = "\n".join(results).strip()

    logger.info(f"📝 Extracted {len(extracted_text)} characters from image.")
    return extracted_text


# -------------------- Quick CLI test --------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    text = extract_text_from_image(image_bytes)
    print("=" * 60)
    print("  EXTRACTED TEXT")
    print("=" * 60)
    print(text)
    print("=" * 60)
