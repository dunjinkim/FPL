"""
Rod segmentation and detection from SEM images.

Pipeline:
1. Mask out the SEM info strip at the bottom
2. Denoise with Gaussian blur
3. Otsu binarisation (rods are bright on dark background; auto-invert if needed)
4. Morphological opening to remove salt-and-pepper noise
5. Watershed to separate touching rods
6. Connected-component filtering by area and aspect ratio
"""

import cv2
import numpy as np
from skimage import measure, morphology, segmentation, feature


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _mask_strip(gray: np.ndarray, strip_y: int) -> np.ndarray:
    masked = gray.copy()
    masked[strip_y:, :] = 0
    return masked


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold; auto-invert so rods are white."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background is white (majority of pixels are white) → invert
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _morphological_clean(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return opened


def _watershed_separate(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Use distance-transform + watershed to split touching objects.
    Returns a label image (0 = background).
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Suppress distance values in the info strip region (already zeroed in gray)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    label_img = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    return label_img


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

    # Convexity defects
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


def detect_rods(
    image: np.ndarray,
    strip_y: int,
    min_area_px: int = 200,
    min_aspect_ratio: float = 1.5,
    max_area_px: int | None = None,
) -> list[dict]:
    """
    Detect rod-shaped objects in a SEM image.

    Parameters
    ----------
    image        : BGR or grayscale SEM image
    strip_y      : y-coordinate where the SEM info strip begins (mask below this)
    min_area_px  : minimum object area in pixels
    min_aspect_ratio : minimum length/diameter ratio to be considered a rod
    max_area_px  : maximum object area (auto-set to 5% of image area if None)

    Returns
    -------
    List of dicts, one per detected candidate, with shape features and contour.
    Each dict also contains an 'image_h' and 'image_w' key for boundary checks.
    """
    gray = _to_gray(image)
    h, w = gray.shape

    if max_area_px is None:
        max_area_px = int(h * w * 0.05)

    masked = _mask_strip(gray, strip_y)
    binary = _binarise(masked)
    cleaned = _morphological_clean(binary)

    # Watershed label map
    labels = _watershed_separate(cleaned, masked)

    # Also fall back to simple connected components on the cleaned binary
    # (watershed can sometimes merge objects)
    cc_labels, _ = cv2.connectedComponents(cleaned)

    # Use the label map with more objects (typically watershed gives more)
    n_watershed = labels.max()
    n_cc = cc_labels.max()
    use_labels = labels if n_watershed >= n_cc else cc_labels

    rods = []
    for label_id in range(1, use_labels.max() + 1):
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

        # Intensity features from the original (un-masked) gray image
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
