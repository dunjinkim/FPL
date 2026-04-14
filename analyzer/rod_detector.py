"""
Rod segmentation and detection from SEM images.

Supports two fundamentally different paradigms:

INTENSITY-BASED (works when rods are brighter/darker than background):
  otsu        — classic global Otsu threshold
  multi_otsu  — 3-class Otsu; tolerates low contrast
  adaptive    — local Gaussian threshold; handles uneven illumination
  triangle    — one-sided histogram peak

EDGE/SHAPE-BASED (works even when brightness inside ≈ background):
  edge_fill         — Canny outlines → morphological close → flood-fill interior
  gradient_watershed— Sobel gradient image used as watershed terrain;
                      low-gradient interiors become labelled regions

auto — tries intensity methods first; if few objects found, falls back to
       edge_fill (recommended for low-contrast SEM images)
"""

import cv2
import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.filters import threshold_multiotsu, threshold_triangle, sobel
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _mask_strip(gray: np.ndarray, strip_y: int) -> np.ndarray:
    masked = gray.copy()
    masked[strip_y:, :] = 0
    return masked


# ── Contrast enhancement ──────────────────────────────────────────────────────

def _enhance_contrast(gray: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """CLAHE — boosts local contrast independently in each tile."""
    h, w = gray.shape
    tile = int(np.clip(min(h, w) // 8, 8, 64))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _unsharp_mask(gray: np.ndarray, sigma: int = 5, amount: float = 1.5) -> np.ndarray:
    """
    Unsharp masking: subtracts a blurred version to emphasise local differences.
    Useful before edge detection — makes rod boundaries stand out more.
    """
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    sharpened = gray.astype(np.float32) * (1 + amount) - blurred * amount
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(gray: np.ndarray, enhance_contrast: bool, clahe_clip: float) -> np.ndarray:
    """CLAHE (optional) → bilateral filter (edge-preserving denoising)."""
    if enhance_contrast:
        gray = _enhance_contrast(gray, clip_limit=clahe_clip)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised


# ── Shared utility ────────────────────────────────────────────────────────────

def _ensure_rods_white(binary: np.ndarray) -> np.ndarray:
    """Invert so rods = 255 (foreground white)."""
    if np.mean(binary) > 127:
        return cv2.bitwise_not(binary)
    return binary


def _count_blobs(binary: np.ndarray) -> int:
    """Quick count of connected components (for auto method selection)."""
    _, n = cv2.connectedComponents(binary)
    return n - 1  # subtract background


# ── INTENSITY-BASED methods ───────────────────────────────────────────────────

def _do_otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _ensure_rods_white(binary)


def _do_adaptive(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    block = max(int(min(h, w) * 0.05) | 1, 11)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -5
    )
    return _ensure_rods_white(binary)


def _try_multi_otsu(gray: np.ndarray) -> np.ndarray | None:
    if not _SKIMAGE_OK:
        return None
    try:
        thresholds = threshold_multiotsu(gray, classes=3)
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


# ── EDGE/SHAPE-BASED methods ──────────────────────────────────────────────────

def _do_edge_fill(gray: np.ndarray) -> np.ndarray:
    """
    Detect rods by their OUTLINES rather than brightness.

    Works even when rod interior brightness ≈ background brightness.

    Steps
    -----
    1. Unsharp masking  → amplifies subtle intensity gradients at rod surfaces
    2. Canny            → detects the actual outlines (edges)
    3. Morphological CLOSE → seals small gaps in the outline so rods are closed shapes
    4. Flood-fill background (from image border) → only the INTERIOR of closed
       outlines remains white (= rod regions)
    5. Remove small filled blobs (noise / dust)
    """
    # 1. Enhance local differences before edge detection
    sharpened = _unsharp_mask(gray, sigma=3, amount=1.5)

    # 2. Canny: use automatic threshold (median-based rule)
    median = float(np.median(sharpened))
    lo = max(0,   int(0.5 * median))
    hi = min(255, int(1.5 * median))
    edges = cv2.Canny(sharpened, lo, hi)

    # 3. Close gaps: dilate then erode so outlines become solid closed curves
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5, iterations=3)
    closed = cv2.dilate(closed, k5, iterations=1)

    # 4. Flood-fill: treat closed edges as walls.
    #    Start from the image border → fills background = 0.
    #    Enclosed areas (rod interiors) cannot be reached → stay 255.
    h, w = closed.shape
    walls = cv2.bitwise_not(closed)          # edges = 0 (walls), open space = 255
    # pad border so flood-fill always starts outside
    padded = cv2.copyMakeBorder(walls, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 0)
    filled = padded[1:-1, 1:-1]              # remove padding

    # 5. Remove tiny noise blobs (< 50 px²)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k3, iterations=1)
    return cleaned


def _do_gradient_watershed(gray: np.ndarray) -> np.ndarray | None:
    """
    Watershed on the Sobel gradient image.

    The gradient is HIGH at rod boundaries and LOW inside rods.
    Watershed finds the "valleys" (rod interiors) as separate labelled regions.

    Steps
    -----
    1. Compute Sobel gradient magnitude
    2. Distance transform of (inverted gradient) → peaks = rod centres
    3. Mark peaks as seeds; run watershed
    4. Return binary of all labelled regions
    """
    if not _SKIMAGE_OK:
        return None
    try:
        # Gradient magnitude (high at edges)
        gradient = sobel(gray.astype(float) / 255.0)

        # Distance transform of inverted gradient: peaks at rod centres
        inv_grad = 1.0 - gradient / (gradient.max() + 1e-9)
        distance = ndi.distance_transform_edt(inv_grad > 0.3)

        # Seed points at local maxima of distance map
        h, w = gray.shape
        min_sep = max(int(min(h, w) * 0.02), 5)
        coords = peak_local_max(distance, min_distance=min_sep, labels=(inv_grad > 0.3))
        seed_mask = np.zeros_like(gray, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(seed_mask)

        # Watershed on gradient
        labels = watershed(gradient, markers, mask=(gradient < 0.25))
        binary = np.where(labels > 0, 255, 0).astype(np.uint8)
        return binary
    except Exception:
        return None


# ── Smart dispatcher ──────────────────────────────────────────────────────────

def _binarise(gray: np.ndarray, method: str = "auto") -> np.ndarray:
    """
    Produce a binary image (foreground = 255) using the chosen strategy.

    method options
    --------------
    auto              — smart cascade (see below)
    otsu              — global Otsu (intensity-based)
    multi_otsu        — 3-class Otsu (intensity-based, tolerates low contrast)
    adaptive          — local adaptive threshold (intensity-based)
    triangle          — triangle threshold (intensity-based)
    edge_fill         — Canny outline detection + fill (SHAPE-BASED)
    gradient_watershed— Sobel gradient watershed (SHAPE-BASED)

    auto cascade
    ------------
    1. Try multi_otsu  → use if ≥ 3 objects found
    2. Try edge_fill   → use if ≥ 3 objects found
    3. Fall back to adaptive
    """
    if method == "otsu":
        return _do_otsu(gray)
    if method == "multi_otsu":
        return _try_multi_otsu(gray) or _do_adaptive(gray)
    if method == "adaptive":
        return _do_adaptive(gray)
    if method == "triangle":
        return _do_triangle(gray) or _do_otsu(gray)
    if method == "edge_fill":
        return _do_edge_fill(gray)
    if method == "gradient_watershed":
        return _do_gradient_watershed(gray) or _do_edge_fill(gray)

    # ── auto ──
    b_mo = _try_multi_otsu(gray)
    if b_mo is not None and _count_blobs(b_mo) >= 3:
        return b_mo

    b_ef = _do_edge_fill(gray)
    if _count_blobs(b_ef) >= 3:
        return b_ef

    return _do_adaptive(gray)


# ── Morphology ────────────────────────────────────────────────────────────────

def _morphological_clean(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)


# ── Watershed separation ──────────────────────────────────────────────────────

def _watershed_separate(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
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
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    long_side  = max(rw, rh)
    short_side = min(rw, rh)
    aspect_ratio = long_side / short_side if short_side > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    bx, by, bw, bh = cv2.boundingRect(contour)
    extent = area / (bw * bh) if bw * bh > 0 else 0
    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0

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
        "area_px": area, "perimeter": perimeter,
        "long_side_px": long_side, "short_side_px": short_side,
        "aspect_ratio": aspect_ratio, "solidity": solidity,
        "extent": extent, "circularity": circularity,
        "convexity_defect_count": defect_count,
        "center_x": cx, "center_y": cy, "angle": angle,
        "rect": rect, "contour": contour,
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
    enhance_contrast  : apply CLAHE before thresholding
    clahe_clip        : CLAHE clip limit
    threshold_method  : 'auto' | 'otsu' | 'adaptive' | 'multi_otsu' |
                        'triangle' | 'edge_fill' | 'gradient_watershed'
    """
    gray = _to_gray(image)
    h, w = gray.shape

    if max_area_px is None:
        max_area_px = int(h * w * 0.05)

    masked      = _mask_strip(gray, strip_y)
    preprocessed = _preprocess(masked, enhance_contrast, clahe_clip)
    binary      = _binarise(preprocessed, method=threshold_method)
    cleaned     = _morphological_clean(binary)

    labels      = _watershed_separate(cleaned, preprocessed)
    _, cc_labels = cv2.connectedComponents(cleaned)

    n_ws = int(labels.max())
    n_cc = int(cc_labels.max())
    use_labels = labels if n_ws >= n_cc else cc_labels

    rods = []
    for label_id in range(1, int(use_labels.max()) + 1):
        mask = np.uint8(use_labels == label_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        feats = _extract_contour_features(contour)
        area  = feats["area_px"]

        if area < min_area_px or area > max_area_px:
            continue
        if feats["aspect_ratio"] < min_aspect_ratio:
            continue

        obj_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(obj_mask, [contour], -1, 255, -1)
        pixels = gray[obj_mask == 255]
        feats["mean_intensity"] = float(pixels.mean()) if len(pixels) else 0.0
        feats["std_intensity"]  = float(pixels.std())  if len(pixels) else 0.0
        feats["image_h"] = h
        feats["image_w"] = w
        feats["label_id"] = label_id
        rods.append(feats)

    return rods
