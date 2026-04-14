"""
Rod segmentation and detection from SEM images.

Pipeline:
1. Mask out the SEM info strip at the bottom
2. (Optional) CLAHE contrast enhancement for low-contrast images
3. Bilateral denoising — edge-preserving, keeps rod boundaries sharp
4. Thresholding (auto / otsu / adaptive / multi_otsu / triangle)
5. Morphological opening to remove noise
6. Watershed to separate touching rods
7. Connected-component filtering by area and aspect ratio
"""

import cv2
import numpy as np

try:
    from skimage.filters import threshold_multiotsu, threshold_triangle
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _mask_strip(gray: np.ndarray, strip_y: int) -> np.ndarray:
    masked = gray.copy()
    masked[strip_y:, :] = 0
    return masked


# ── Contrast enhancement ─────────────────────────────────────────────────────

def _enhance_contrast(gray: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Splits the image into a grid of tiles and equalises each tile's histogram
    independently, so local contrast is boosted even when global contrast is low.

    Parameters
    ----------
    gray       : uint8 grayscale image
    clip_limit : higher → stronger enhancement (1.0 = mild, 6.0 = aggressive)
    """
    h, w = gray.shape
    # tile size ≈ 1/8 of shorter dimension, clamped to [8, 64]
    tile = int(np.clip(min(h, w) // 8, 8, 64))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(gray: np.ndarray, enhance_contrast: bool, clahe_clip: float) -> np.ndarray:
    """CLAHE → bilateral filter (edge-preserving denoising)."""
    if enhance_contrast:
        gray = _enhance_contrast(gray, clip_limit=clahe_clip)
    # bilateralFilter: d=9, sigmaColor/sigmaSpace=75
    # Blurs smoothly within uniform regions but preserves sharp rod edges
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


# ── Thresholding ──────────────────────────────────────────────────────────────

def _ensure_rods_white(binary: np.ndarray) -> np.ndarray:
    """Invert if background (majority) ended up white."""
    if np.mean(binary) > 127:
        return cv2.bitwise_not(binary)
    return binary


def _binarise(gray: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Threshold *gray* into a binary image where foreground (rods) = 255.

    Parameters
    ----------
    method : 'auto' | 'otsu' | 'adaptive' | 'multi_otsu' | 'triangle'

    'auto'      → tries multi_otsu first; falls back to adaptive
    'otsu'      → classic single global Otsu
    'adaptive'  → local adaptive threshold (best for uneven illumination)
    'multi_otsu'→ 3-class Otsu; rod pixels are the brightest class
    'triangle'  → good when histogram has one tall peak + long tail
    """
    if method == "auto":
        binary = _try_multi_otsu(gray)
        if binary is None:
            binary = _do_adaptive(gray)
        return binary

    if method == "multi_otsu":
        binary = _try_multi_otsu(gray)
        return binary if binary is not None else _do_adaptive(gray)

    if method == "adaptive":
        return _do_adaptive(gray)

    if method == "triangle":
        binary = _do_triangle(gray)
        return binary if binary is not None else _do_otsu(gray)

    # default: otsu
    return _do_otsu(gray)


def _do_otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _ensure_rods_white(binary)


def _do_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Local adaptive threshold with a large block window.
    C = -5 biases slightly toward brighter pixels being foreground.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # block size must be odd; use ~5% of shorter image dimension
    h, w = gray.shape
    block = max(int(min(h, w) * 0.05) | 1, 11)  # ensure odd, ≥ 11
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, -5,
    )
    return _ensure_rods_white(binary)


def _try_multi_otsu(gray: np.ndarray) -> np.ndarray | None:
    """
    3-class Multi-Otsu: background | rods | bright highlights.
    Returns binary (rods+highlights = 255) or None if skimage not available.
    """
    if not _SKIMAGE_OK:
        return None
    try:
        thresholds = threshold_multiotsu(gray, classes=3)
        # Use the lower threshold: everything above background → rod
        binary = np.where(gray >= thresholds[0], 255, 0).astype(np.uint8)
        return _ensure_rods_white(binary)
    except Exception:
        return None


def _do_triangle(gray: np.ndarray) -> np.ndarray | None:
    if not _SKIMAGE_OK:
        return None
    try:
        thresh = threshold_triangle(gray)
        binary = np.where(gray > thresh, 255, 0).astype(np.uint8)
        return _ensure_rods_white(binary)
    except Exception:
        return None


# ── Morphology ────────────────────────────────────────────────────────────────

def _morphological_clean(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return opened


# ── Watershed ────────────────────────────────────────────────────────────────

def _watershed_separate(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Distance-transform + watershed to split touching objects.
    Returns a label image (0 = background).
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist == 0:
        return np.zeros_like(binary, dtype=np.int32)

    _, sure_fg = cv2.threshold(dist, 0.35 * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    return np.where(markers > 1, markers - 1, 0).astype(np.int32)


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_contour_features(contour) -> dict:
    """Compute shape features from an OpenCV contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    long_side = max(w, h)
    short_side = min(w, h)
    aspect_ratio = long_side / short_side if short_side > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    bx, by, bw, bh = cv2.boundingRect(contour)
    extent = area / (bw * bh) if bw * bh > 0 else 0

    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    defect_count = 0
    if hull_idx is not None and len(hull_idx) > 3 and len(contour) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                defect_count = len(defects)
        except cv2.error:
            pass

    return {
        "area_px": area,
        "perimeter": perimeter,
        "long_side_px": long_side,
        "short_side_px": short_side,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "circularity": circularity,
        "convexity_defect_count": defect_count,
        "center_x": cx,
        "center_y": cy,
        "angle": angle,
        "rect": rect,
        "contour": contour,
        "bbox": (bx, by, bw, bh),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def detect_rods(
    image: np.ndarray,
    strip_y: int,
    min_area_px: int = 200,
    min_aspect_ratio: float = 1.5,
    max_area_px: int | None = None,
    enhance_contrast: bool = True,
    clahe_clip: float = 3.0,
    threshold_method: str = "auto",
) -> list[dict]:
    """
    Detect rod-shaped objects in a SEM image.

    Parameters
    ----------
    image             : BGR or grayscale SEM image
    strip_y           : y-coordinate where the SEM info strip begins
    min_area_px       : minimum object area in pixels
    min_aspect_ratio  : minimum length/diameter ratio
    max_area_px       : maximum object area (default 5 % of image)
    enhance_contrast  : apply CLAHE before thresholding (helps low-contrast images)
    clahe_clip        : CLAHE clip limit (1.0 mild → 6.0 aggressive)
    threshold_method  : 'auto' | 'otsu' | 'adaptive' | 'multi_otsu' | 'triangle'

    Returns
    -------
    List of feature dicts, one per detected rod candidate.
    """
    gray = _to_gray(image)
    h, w = gray.shape

    if max_area_px is None:
        max_area_px = int(h * w * 0.05)

    masked = _mask_strip(gray, strip_y)
    preprocessed = _preprocess(masked, enhance_contrast, clahe_clip)
    binary = _binarise(preprocessed, method=threshold_method)
    cleaned = _morphological_clean(binary)

    # Watershed label map
    labels = _watershed_separate(cleaned, preprocessed)

    # Also compute simple connected components as fallback
    _, cc_labels = cv2.connectedComponents(cleaned)

    # Keep whichever produces more objects
    n_watershed = int(labels.max())
    n_cc = int(cc_labels.max())
    use_labels = labels if n_watershed >= n_cc else cc_labels

    rods = []
    for label_id in range(1, int(use_labels.max()) + 1):
        mask = np.uint8(use_labels == label_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        feats = _extract_contour_features(contour)
        area = feats["area_px"]

        if area < min_area_px or area > max_area_px:
            continue
        if feats["aspect_ratio"] < min_aspect_ratio:
            continue

        # Intensity features from the original (pre-CLAHE) gray image
        obj_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(obj_mask, [contour], -1, 255, -1)
        pixels = gray[obj_mask == 255]
        feats["mean_intensity"] = float(pixels.mean()) if len(pixels) else 0.0
        feats["std_intensity"] = float(pixels.std()) if len(pixels) else 0.0

        feats["image_h"] = h
        feats["image_w"] = w
        feats["label_id"] = label_id

        rods.append(feats)

    return rods
