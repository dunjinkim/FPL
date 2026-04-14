"""
Scale bar detection and nm/pixel ratio calculation for SEM images.

Strategy:
1. Crop the bottom info-strip of the image (where SEM metadata lives)
2. Detect the horizontal scale bar line via HoughLinesP
3. OCR the strip to find a string like "500 nm" or "1.00 μm"
4. Return nm_per_pixel ratio and metadata for display
"""

import re
import cv2
import numpy as np

try:
    import easyocr
    _reader = None  # lazy init to avoid slow startup

    def _get_reader():
        global _reader
        if _reader is None:
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _reader

except ImportError:
    _get_reader = None


# ── Unit normalisation ───────────────────────────────────────────────────────

_UNIT_MAP = {
    "nm": 1.0,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "mm": 1_000_000.0,
}

_SCALE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(nm|µm|μm|um|mm)",
    re.IGNORECASE,
)


def _parse_scale_text(text: str) -> float | None:
    """Return scale value in nm, or None if not found."""
    # normalise common OCR mistakes
    text = text.replace(",", ".").replace("urn", "um").replace("pum", "um")
    m = _SCALE_RE.search(text)
    if m is None:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower().replace("µ", "μ")
    multiplier = _UNIT_MAP.get(unit, None)
    if multiplier is None:
        return None
    return value * multiplier


# ── Scale bar line detection ─────────────────────────────────────────────────

def _detect_scalebar_line(strip: np.ndarray) -> int | None:
    """
    Find the pixel length of the scale bar line in *strip*.

    Returns the pixel length of the longest near-horizontal line, or None.
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip.copy()

    # Try both bright-on-dark and dark-on-bright
    results = []
    for thresh_img in [gray, cv2.bitwise_not(gray)]:
        _, binary = cv2.threshold(thresh_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=strip.shape[1] // 20,
            maxLineGap=5,
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < 5 or angle > 175:  # near-horizontal
                    length = abs(x2 - x1)
                    results.append(length)

    return max(results) if results else None


# ── Public API ───────────────────────────────────────────────────────────────

def detect_scale_bar(
    image: np.ndarray,
    info_strip_ratio: float = 0.15,
) -> dict:
    """
    Detect the scale bar in a SEM image.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale image.
    info_strip_ratio : float
        Fraction of image height reserved for the SEM info strip at the bottom.

    Returns
    -------
    dict with keys:
        nm_per_pixel : float or None
        scale_nm     : float or None   – labelled value in nm
        bar_px       : int or None     – detected bar length in pixels
        text         : str             – raw OCR text from the strip
        strip_y      : int             – y-coordinate where the strip starts
        success      : bool
    """
    h, w = image.shape[:2]
    strip_y = int(h * (1.0 - info_strip_ratio))
    strip = image[strip_y:, :]

    # Convert to BGR for EasyOCR if grayscale
    if strip.ndim == 2:
        strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    else:
        strip_bgr = strip

    # OCR
    ocr_text = ""
    scale_nm = None
    if _get_reader is not None:
        try:
            reader = _get_reader()
            results = reader.readtext(strip_bgr, detail=0, paragraph=True)
            ocr_text = " ".join(results)
            scale_nm = _parse_scale_text(ocr_text)
        except Exception:
            pass

    # Scale bar line length
    bar_px = _detect_scalebar_line(strip_bgr)

    nm_per_pixel = None
    if scale_nm is not None and bar_px is not None and bar_px > 0:
        nm_per_pixel = scale_nm / bar_px

    return {
        "nm_per_pixel": nm_per_pixel,
        "scale_nm": scale_nm,
        "bar_px": bar_px,
        "text": ocr_text,
        "strip_y": strip_y,
        "success": nm_per_pixel is not None,
    }
