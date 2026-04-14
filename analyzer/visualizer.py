"""
Visualisation utilities: annotated result image and distribution plots.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO


# ── Colour scheme ────────────────────────────────────────────────────────────

_COLOR_SINGLE  = (34,  197,  94)   # green  (BGR)
_COLOR_OVERLAP = (239,  68,  68)   # red    (BGR)
_COLOR_PARTIAL = ( 59, 130, 246)   # blue   (BGR)
_COLOR_NOT_ROD = (168,  85, 247)   # purple (BGR)
_COLOR_UNKNOWN = (251, 191,  36)   # amber  (BGR)

_LABEL_COLORS = {
    "single":  _COLOR_SINGLE,
    "overlap": _COLOR_OVERLAP,
    "partial": _COLOR_PARTIAL,
    "not_rod": _COLOR_NOT_ROD,
}


def _label_color(label: str) -> tuple[int, int, int]:
    return _LABEL_COLORS.get(label, _COLOR_UNKNOWN)


# ── Annotated image ──────────────────────────────────────────────────────────

def annotate_image(
    image: np.ndarray,
    rods: list[dict],
    df: pd.DataFrame,
    show_measurements: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of *image*.

    Parameters
    ----------
    image : original BGR or grayscale image
    rods  : list of feature dicts from rod_detector (contains 'contour', 'rect')
    df    : DataFrame from measurer (must have 'overlap_label', '_rod_ref')
    show_measurements : if True, print length×diameter on each rod

    Returns
    -------
    Annotated BGR image (numpy array).
    """
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()

    # Build a mapping: rod index → row
    idx_to_row = {int(row["_rod_ref"]): row for _, row in df.iterrows()}

    for rod_idx, rod in enumerate(rods):
        row = idx_to_row.get(rod_idx)
        if row is None:
            continue

        label = row.get("overlap_label", "unknown")
        color = _label_color(label)
        rid = int(row["id"])

        # Draw minimum bounding rectangle
        box = cv2.boxPoints(rod["rect"])
        box = np.intp(box)
        cv2.drawContours(annotated, [box], 0, color, 2)

        # Label position: top-left corner of bounding rect
        cx, cy = rod["rect"][0]
        text_x = max(int(cx) - 30, 0)
        text_y = max(int(cy) - 8, 12)

        cv2.putText(
            annotated, f"#{rid}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

        if show_measurements and label == "single":
            meas = f"L:{row['length_nm']:.0f} D:{row['diameter_nm']:.0f}"
            cv2.putText(
                annotated, meas,
                (text_x, text_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )

    return annotated


def annotate_scale_bar(image: np.ndarray, scale_info: dict) -> np.ndarray:
    """Draw a highlight around the detected scale bar region."""
    annotated = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    strip_y = scale_info.get("strip_y", image.shape[0])
    h = image.shape[0]
    cv2.rectangle(annotated, (0, strip_y), (image.shape[1], h), (255, 215, 0), 2)
    return annotated


# ── Distribution plots ────────────────────────────────────────────────────────

def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def plot_distributions(df: pd.DataFrame) -> dict[str, bytes]:
    """
    Generate histogram + KDE plots for length, diameter, and aspect ratio.

    Parameters
    ----------
    df : DataFrame filtered to single rods only.

    Returns
    -------
    Dict of {metric_name: PNG bytes}.
    """
    plots = {}
    sns.set_theme(style="whitegrid", palette="muted")

    for col, label, unit in [
        ("length_nm", "Length", "nm"),
        ("diameter_nm", "Diameter", "nm"),
        ("aspect_ratio", "Aspect Ratio", ""),
    ]:
        vals = df[col].dropna()
        if vals.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(vals, kde=True, ax=ax, color="#3b82f6", edgecolor="white", linewidth=0.5)
        ax.axvline(vals.mean(), color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean: {vals.mean():.1f}")
        ax.axvline(vals.median(), color="#f97316", linestyle=":", linewidth=1.5, label=f"Median: {vals.median():.1f}")
        xlabel = f"{label} ({unit})" if unit else label
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{label} Distribution  (n={len(vals)})", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()

        plots[col] = _fig_to_png_bytes(fig)

    return plots


def plot_scatter(df: pd.DataFrame) -> bytes:
    """Length vs Diameter scatter plot."""
    vals = df[["length_nm", "diameter_nm"]].dropna()
    if vals.empty:
        return b""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(vals["diameter_nm"], vals["length_nm"], alpha=0.6, s=20, color="#6366f1")
    ax.set_xlabel("Diameter (nm)", fontsize=11)
    ax.set_ylabel("Length (nm)", fontsize=11)
    ax.set_title("Length vs Diameter", fontsize=12)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ── Thumbnail extraction ─────────────────────────────────────────────────────

def extract_thumbnail(
    image: np.ndarray,
    rod: dict,
    padding: int = 10,
    size: tuple[int, int] = (80, 80),
) -> np.ndarray:
    """
    Crop a small patch around a rod for display in the labelling UI.

    Returns an RGB image of `size` pixels.
    """
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()

    bx, by, bw, bh = rod["bbox"]
    h_img, w_img = img.shape[:2]

    x1 = max(bx - padding, 0)
    y1 = max(by - padding, 0)
    x2 = min(bx + bw + padding, w_img)
    y2 = min(by + bh + padding, h_img)

    crop = img[y1:y2, x1:x2]
    thumb = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
