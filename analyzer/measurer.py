"""
Convert pixel-space rod measurements to real-world units (nm).
"""

import numpy as np
import pandas as pd


def measure_rods(
    rods: list[dict],
    nm_per_pixel: float,
    exclude_boundary: bool = True,
) -> pd.DataFrame:
    """
    Compute length and diameter for each detected rod.

    Parameters
    ----------
    rods          : list of feature dicts from rod_detector.detect_rods()
    nm_per_pixel  : calibration factor (nm per pixel)
    exclude_boundary : if True, skip rods that touch the image border

    Returns
    -------
    DataFrame with columns:
        id, length_nm, diameter_nm, aspect_ratio,
        center_x, center_y, area_px2,
        solidity, circularity,
        is_boundary, overlap_label (placeholder, filled later)
    """
    records = []
    for i, rod in enumerate(rods):
        h, w = rod["image_h"], rod["image_w"]
        bx, by, bw, bh = rod["bbox"]

        is_boundary = (
            bx <= 0 or by <= 0 or (bx + bw) >= w or (by + bh) >= h
        )
        if exclude_boundary and is_boundary:
            continue

        length_nm = rod["long_side_px"] * nm_per_pixel
        diameter_nm = rod["short_side_px"] * nm_per_pixel

        records.append(
            {
                "id": i + 1,
                "length_nm": round(length_nm, 2),
                "diameter_nm": round(diameter_nm, 2),
                "aspect_ratio": round(rod["aspect_ratio"], 3),
                "center_x": round(rod["center_x"], 1),
                "center_y": round(rod["center_y"], 1),
                "area_px2": round(rod["area_px"], 1),
                "solidity": round(rod["solidity"], 3),
                "circularity": round(rod["circularity"], 3),
                "mean_intensity": round(rod["mean_intensity"], 2),
                "std_intensity": round(rod["std_intensity"], 2),
                "convexity_defect_count": rod["convexity_defect_count"],
                "is_boundary": is_boundary,
                "overlap_label": "unknown",  # filled by overlap_classifier
                "_rod_ref": i,              # index back to rods list
            }
        )

    return pd.DataFrame(records)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for single (non-overlapping) rods.

    Parameters
    ----------
    df : DataFrame from measure_rods(), with overlap_label column filled.

    Returns
    -------
    DataFrame of statistics (one row per metric).
    """
    single = df[df["overlap_label"] == "single"]
    if single.empty:
        return pd.DataFrame()

    stats = []
    for col in ["length_nm", "diameter_nm", "aspect_ratio"]:
        vals = single[col].dropna()
        stats.append(
            {
                "metric": col,
                "count": len(vals),
                "mean": round(vals.mean(), 2),
                "std": round(vals.std(), 2),
                "median": round(vals.median(), 2),
                "min": round(vals.min(), 2),
                "max": round(vals.max(), 2),
            }
        )
    return pd.DataFrame(stats)
