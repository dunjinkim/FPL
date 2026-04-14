"""
SEM Rod Analyzer — Streamlit Web Application

Usage:
    streamlit run app.py
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from analyzer.scale_bar import detect_scale_bar
from analyzer.rod_detector import detect_rods, _do_edge_fill, _extract_contour_features
from analyzer.measurer import measure_rods, compute_statistics
from analyzer.overlap_classifier import (
    OverlapClassifier,
    LABEL_SINGLE, LABEL_OVERLAP, LABEL_PARTIAL, LABEL_NOT_ROD,
    ALL_LABELS,
)
from analyzer.visualizer import (
    annotate_image,
    annotate_scale_bar,
    plot_distributions,
    plot_scatter,
    extract_thumbnail,
)

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEM Rod Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ───────────────────────────────────────────────────
def _init_state():
    defaults = {
        "image": None,          # np.ndarray BGR
        "image_name": "",
        "scale_info": None,
        "rods": None,
        "df_measured": None,
        "label_pending": {},
        "manual_rods": [],      # list[dict] — manually annotated rod features
        "manual_df": None,      # pd.DataFrame of manual measurements
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helper: load classifier ──────────────────────────────────────────────────
@st.cache_resource
def get_classifier():
    return OverlapClassifier(
        model_path=ROOT / "models" / "overlap_model.pkl",
        labels_path=ROOT / "training_data" / "labels.csv",
    )


# ── Manual annotation helper ─────────────────────────────────────────────────
def _process_manual_rect(img_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> dict | None:
    """
    Extract rod features from a user-drawn bounding box.

    1. Crop the region and attempt edge-fill segmentation inside it.
    2. If a contour is found, use its minAreaRect for precise measurement.
    3. Fallback: treat the drawn rectangle itself as the rod shape.
    """
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0 or min(crop.shape[:2]) < 5:
        return None

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

    feats = None
    try:
        binary = _do_edge_fill(gray_crop)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 50:
                feats = _extract_contour_features(contour)
    except Exception:
        pass

    if feats is None:
        # Fallback: use the drawn rectangle
        local_contour = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)
        feats = _extract_contour_features(local_contour)

    # Offset position-dependent fields to original image coordinates
    feats["center_x"] += x
    feats["center_y"] += y
    rc, rs, ra = feats["rect"]
    feats["rect"] = ((rc[0] + x, rc[1] + y), rs, ra)
    bx, by, bw, bh = feats["bbox"]
    feats["bbox"] = (bx + x, by + y, bw, bh)
    if feats.get("contour") is not None:
        feats["contour"] = feats["contour"] + np.array([[[x, y]]])

    feats["image_h"] = img_bgr.shape[0]
    feats["image_w"] = img_bgr.shape[1]
    feats["label_id"] = -1
    feats["mean_intensity"] = float(gray_crop.mean())
    feats["std_intensity"] = float(gray_crop.std())
    return feats


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 SEM Rod Analyzer")
    st.markdown("---")

    uploaded = st.file_uploader(
        "SEM 이미지 업로드",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="TIF/TIFF, PNG, JPG, BMP 모두 지원합니다.",
    )

    # ── Store image immediately on upload ────────────────────────────────────
    if uploaded is not None:
        if st.session_state.image_name != uploaded.name:
            file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
            img_loaded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.image = img_loaded
            st.session_state.image_name = uploaded.name
            # Reset all analysis state for new image
            st.session_state.scale_info = None
            st.session_state.rods = None
            st.session_state.df_measured = None
            st.session_state.label_pending = {}
            st.session_state.manual_rods = []
            st.session_state.manual_df = None

    st.markdown("### 분석 파라미터")
    strip_ratio = st.slider(
        "정보 영역 비율 (하단 %)", min_value=5, max_value=30, value=7, step=1,
        help="SEM 메타데이터(스케일바) 영역이 이미지 하단 몇 %를 차지하는지 설정합니다.",
    ) / 100.0

    min_area = st.slider(
        "최소 라드 면적 (px²)", min_value=50, max_value=2000, value=300, step=50,
        help="이 면적보다 작은 객체는 라드로 인식하지 않습니다.",
    )
    min_ar = st.slider(
        "최소 종횡비", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
        help="길이/직경 비율이 이 값보다 작으면 구형 입자로 간주합니다.",
    )
    exclude_boundary = st.checkbox("경계 라드 제외", value=True,
        help="이미지 가장자리에 걸친 라드를 측정에서 제외합니다.")

    with st.expander("고급 세그멘테이션 설정"):
        enhance_contrast = st.checkbox(
            "콘트라스트 향상 (CLAHE)", value=True,
            help="저콘트라스트 이미지에서 라드 검출률을 높입니다.",
        )
        clahe_clip = st.slider(
            "CLAHE 강도", min_value=1.0, max_value=6.0, value=3.0, step=0.5,
            disabled=not enhance_contrast,
        )
        threshold_method = st.selectbox(
            "이진화 방법",
            options=["auto", "edge_fill", "gradient_watershed",
                     "multi_otsu", "adaptive", "otsu", "triangle"],
            index=0,
            format_func=lambda x: {
                "auto":               "자동 (권장)",
                "edge_fill":          "엣지 채우기 ★ — 밝기 무관, 형태로 인식",
                "gradient_watershed": "기울기 Watershed — 형태 기반",
                "multi_otsu":         "Multi-Otsu — 저콘트라스트에 강함",
                "adaptive":           "적응형 — 조명 불균일에 강함",
                "otsu":               "Otsu — 기본",
                "triangle":           "Triangle — 히스토그램 한쪽 치우침",
            }[x],
            help="라드와 배경 밝기가 비슷하면 '엣지 채우기'를 선택하세요.",
        )

    st.markdown("---")
    st.markdown("### 분류 모델")
    clf = get_classifier()
    counts = clf.label_counts()
    total = sum(counts.values())
    label_names = {"single": "단일", "overlap": "겹침", "partial": "일부", "not_rod": "라드 아님"}
    st.markdown(f"학습 데이터: **{total}개**")
    for lbl, kor in label_names.items():
        st.markdown(f"- {kor}: {counts[lbl]}개")
    st.markdown(f"현재 모드: `{clf.mode}`" +
                (f" (정확도 {clf.accuracy*100:.1f}%)" if clf.accuracy else ""))

    if clf.can_train():
        if st.button("모델 재학습", type="primary"):
            with st.spinner("학습 중..."):
                acc = clf.train()
            st.success(f"학습 완료! 정확도: {acc*100:.1f}%")
            st.cache_resource.clear()
            st.rerun()
    else:
        qualified = sum(1 for v in counts.values() if v >= 5)
        st.caption(f"모델 학습까지 최소 2개 클래스 각 5개 이상 필요 (현재 충족: {qualified}개 클래스)")

    st.markdown("---")
    if st.button("분석 실행", type="primary",
                 disabled=st.session_state.image is None,
                 use_container_width=True):
        img_bgr_run = st.session_state.image
        clf_run = get_classifier()

        with st.spinner("스케일바 검출 중..."):
            scale_info_run = detect_scale_bar(img_bgr_run, info_strip_ratio=strip_ratio)
        st.session_state.scale_info = scale_info_run

        if not scale_info_run["success"]:
            st.warning("스케일바를 자동으로 검출하지 못했습니다. Tab 1에서 nm/px를 직접 입력해 주세요.")

        with st.spinner("라드 검출 중..."):
            rods_run = detect_rods(
                img_bgr_run,
                strip_y=scale_info_run["strip_y"],
                min_area_px=min_area,
                min_aspect_ratio=min_ar,
                enhance_contrast=enhance_contrast,
                clahe_clip=clahe_clip,
                threshold_method=threshold_method,
            )
        st.session_state.rods = rods_run

        if scale_info_run["success"] and rods_run:
            with st.spinner("측정 및 분류 중..."):
                df_run = measure_rods(rods_run, scale_info_run["nm_per_pixel"],
                                      exclude_boundary=exclude_boundary)
                df_run = clf_run.classify(df_run)
            st.session_state.df_measured = df_run
        elif rods_run:
            st.info("스케일바 정보를 Tab 1에서 입력 후 '측정 적용' 버튼을 눌러 주세요.")
            st.session_state.df_measured = None
        else:
            st.session_state.df_measured = None


# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.image is None:
    st.markdown(
        """
        ## SEM 실리카 라드 자동 측정 프로그램

        **사용 방법**
        1. 왼쪽 사이드바에서 SEM 이미지를 업로드합니다.
        2. 필요시 분석 파라미터를 조정합니다.
        3. **분석 실행** 버튼을 클릭합니다.
        4. 자동 검출이 안 될 경우 **✏️ 수동 라드 표시** 탭에서 직접 표시합니다.

        **지원 형식**: TIF, TIFF, PNG, JPG, BMP
        """
    )
    st.stop()

img_bgr = st.session_state.image
scale_info = st.session_state.scale_info
rods = st.session_state.rods or []
df_all = st.session_state.df_measured
clf = get_classifier()

# ── Image preview before analysis ────────────────────────────────────────────
if scale_info is None:
    st.image(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        caption=st.session_state.image_name,
        use_container_width=True,
    )
    st.info("왼쪽 **[분석 실행]** 버튼을 누르거나, 아래 **✏️ 수동 라드 표시** 탭에서 직접 라드를 표시하세요.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📏 스케일바", "🔍 세그멘테이션", "📊 측정 결과", "🏷️ 학습 데이터", "✏️ 수동 라드 표시"]
)

# ════════════════════════════════════════════════════════
# TAB 1 — Scale bar
# ════════════════════════════════════════════════════════
with tab1:
    if scale_info is None:
        st.info("분석을 먼저 실행하세요.")
    else:
        col_img, col_info = st.columns([2, 1])

        with col_img:
            annotated_sb = annotate_scale_bar(img_bgr, scale_info)
            st.image(
                cv2.cvtColor(annotated_sb, cv2.COLOR_BGR2RGB),
                caption="원본 이미지 (스케일바 영역 하이라이트)",
                use_container_width=True,
            )

        with col_info:
            st.markdown("### 스케일바 검출 결과")
            if scale_info["success"]:
                st.success("자동 검출 성공")
                st.metric("검출된 라벨", scale_info["text"] or "(없음)")
                st.metric("스케일바 픽셀 길이", f"{scale_info['bar_px']} px")
                st.metric("스케일 값", f"{scale_info['scale_nm']:.1f} nm")
                st.metric("nm/pixel", f"{scale_info['nm_per_pixel']:.4f}")
            else:
                st.warning("자동 검출 실패")
                st.markdown("OCR 텍스트:")
                st.code(scale_info.get("text", "(없음)"))

            st.markdown("---")
            st.markdown("### 수동 입력")
            manual_nm_per_px = st.number_input(
                "nm / pixel (직접 입력)", min_value=0.001, max_value=10000.0,
                value=float(scale_info["nm_per_pixel"] or 1.0),
                format="%.4f",
            )
            if st.button("측정 적용"):
                with st.spinner("라드 재검출 중..."):
                    rods_new = detect_rods(
                        img_bgr,
                        strip_y=scale_info["strip_y"],
                        min_area_px=min_area,
                        min_aspect_ratio=min_ar,
                        enhance_contrast=enhance_contrast,
                        clahe_clip=clahe_clip,
                        threshold_method=threshold_method,
                    )
                st.session_state.rods = rods_new
                st.session_state.scale_info["nm_per_pixel"] = manual_nm_per_px
                st.session_state.scale_info["success"] = True
                if rods_new:
                    df_new = measure_rods(rods_new, manual_nm_per_px,
                                         exclude_boundary=exclude_boundary)
                    df_new = clf.classify(df_new)
                    st.session_state.df_measured = df_new
                else:
                    st.session_state.df_measured = None
                st.rerun()


# ════════════════════════════════════════════════════════
# TAB 2 — Segmentation
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 세그멘테이션 결과")
    if scale_info is None:
        st.info("분석을 먼저 실행하세요.")
    else:
        from analyzer.rod_detector import (
            _mask_strip, _enhance_contrast, _preprocess,
            _binarise, _morphological_clean,
        )

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        strip_y = scale_info["strip_y"]
        masked = _mask_strip(gray, strip_y)
        preprocessed = _preprocess(masked, enhance_contrast, clahe_clip)
        binary = _binarise(preprocessed, method=threshold_method)
        cleaned = _morphological_clean(binary)

        if enhance_contrast:
            st.markdown("**전처리 비교**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(masked, caption="원본 (마스크 적용)", use_container_width=True, clamp=True)
            with c2:
                enhanced_vis = _enhance_contrast(masked, clip_limit=clahe_clip)
                st.image(enhanced_vis, caption=f"CLAHE 적용 (강도 {clahe_clip})", use_container_width=True, clamp=True)

        st.markdown("**세그멘테이션 결과**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(cleaned, caption=f"이진화 ({threshold_method})", use_container_width=True, clamp=True)

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for rod in rods:
            box = cv2.boxPoints(rod["rect"])
            box = np.intp(box)
            cv2.drawContours(overlay, [box], 0, (59, 130, 246), 2)

        with col_b:
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption=f"검출된 후보 객체: {len(rods)}개",
                use_container_width=True,
            )
            if len(rods) == 0:
                st.warning("라드가 검출되지 않았습니다. **✏️ 수동 라드 표시** 탭을 이용해 보세요.")


# ════════════════════════════════════════════════════════
# TAB 3 — Measurement results
# ════════════════════════════════════════════════════════
with tab3:
    has_auto   = df_all is not None
    has_manual = bool(st.session_state.manual_rods) and st.session_state.manual_df is not None

    if not has_auto and not has_manual:
        st.info("분석을 실행하거나 **✏️ 수동 라드 표시** 탭에서 라드를 표시하세요.")
    else:
        if has_auto:
            df_single  = df_all[df_all["overlap_label"] == LABEL_SINGLE].copy()
            df_overlap = df_all[df_all["overlap_label"] == LABEL_OVERLAP].copy()
            df_partial = df_all[df_all["overlap_label"] == LABEL_PARTIAL].copy()
            df_notrod  = df_all[df_all["overlap_label"] == LABEL_NOT_ROD].copy()

            st.markdown(
                f"**자동 검출:** {len(df_all)}개 &nbsp;|&nbsp; "
                f"🟢 단일: {len(df_single)}개 &nbsp;|&nbsp; "
                f"🔴 겹침: {len(df_overlap)}개 &nbsp;|&nbsp; "
                f"🔵 일부: {len(df_partial)}개 &nbsp;|&nbsp; "
                f"🟣 라드 아님: {len(df_notrod)}개"
            )
            annotated = annotate_image(img_bgr, rods, df_all, show_measurements=True)

            # Overlay manual rods (gold) on top of auto annotation
            if has_manual:
                for rod in st.session_state.manual_rods:
                    box = cv2.boxPoints(rod["rect"])
                    box = np.intp(box)
                    cv2.drawContours(annotated, [box], 0, (0, 215, 255), 2)

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="🟢 단일 / 🔴 겹침 / 🔵 일부 / 🟣 라드 아님 / 🟡 수동 표시",
                use_container_width=True,
            )

            st.markdown("### 통계 요약")
            stats_df = compute_statistics(df_all)
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            if not df_single.empty:
                st.markdown("### 분포 그래프")
                dist_plots = plot_distributions(df_single)
                scatter_png = plot_scatter(df_single)
                col1, col2 = st.columns(2)
                if "length_nm" in dist_plots:
                    with col1:
                        st.image(dist_plots["length_nm"], caption="길이 분포")
                if "diameter_nm" in dist_plots:
                    with col2:
                        st.image(dist_plots["diameter_nm"], caption="직경 분포")
                col3, col4 = st.columns(2)
                if "aspect_ratio" in dist_plots:
                    with col3:
                        st.image(dist_plots["aspect_ratio"], caption="종횡비 분포")
                if scatter_png:
                    with col4:
                        st.image(scatter_png, caption="길이 vs 직경")

            display_cols = ["id", "length_nm", "diameter_nm", "aspect_ratio",
                            "center_x", "center_y", "area_px2"]
            st.markdown("### 자동 측정 데이터 (단일 라드)")
            if not df_single.empty:
                st.dataframe(df_single[display_cols], use_container_width=True, hide_index=True)

            st.markdown("### 다운로드")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            stem = Path(st.session_state.image_name).stem
            with col_dl1:
                csv_bytes = df_single[display_cols].to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV 다운로드", data=csv_bytes,
                                   file_name=f"{stem}_rods.csv", mime="text/csv")
            with col_dl2:
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    df_single[display_cols].to_excel(writer, sheet_name="Auto", index=False)
                    if has_manual:
                        st.session_state.manual_df.to_excel(writer, sheet_name="Manual", index=False)
                    if not stats_df.empty:
                        stats_df.to_excel(writer, sheet_name="Statistics", index=False)
                excel_buf.seek(0)
                st.download_button("Excel 다운로드", data=excel_buf.read(),
                                   file_name=f"{stem}_rods.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col_dl3:
                _, img_enc = cv2.imencode(".png", annotated)
                st.download_button("주석 이미지 저장", data=img_enc.tobytes(),
                                   file_name=f"{stem}_annotated.png", mime="image/png")

        if has_manual:
            st.markdown("---")
            st.markdown(f"### 수동 표시 라드: {len(st.session_state.manual_rods)}개")
            st.dataframe(st.session_state.manual_df, use_container_width=True, hide_index=True)
            csv_manual = st.session_state.manual_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("수동 측정 CSV 다운로드", data=csv_manual,
                               file_name=f"{Path(st.session_state.image_name).stem}_manual.csv",
                               mime="text/csv")


# ════════════════════════════════════════════════════════
# TAB 4 — Labelling for ML training
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 학습 데이터 수집")
    st.markdown(
        "🟢 **단일** – 온전한 라드 한 개 &nbsp;|&nbsp; "
        "🔴 **겹침** – 두 개 이상 겹친 클러스터 &nbsp;|&nbsp; "
        "🔵 **일부** – 절단되거나 일부만 보이는 라드 &nbsp;|&nbsp; "
        "🟣 **라드 아님** – 구형 입자, 이물질 등"
    )

    if df_all is None:
        st.info("분석을 먼저 실행하세요.")
    else:
        idx_to_row = {int(row["_rod_ref"]): row for _, row in df_all.iterrows()}
        visible_rods = [(i, rod) for i, rod in enumerate(rods) if i in idx_to_row]

        _LABEL_BADGE = {
            LABEL_SINGLE:  ":green[🟢 단일]",
            LABEL_OVERLAP: ":red[🔴 겹침]",
            LABEL_PARTIAL: ":blue[🔵 일부]",
            LABEL_NOT_ROD: ":violet[🟣 라드 아님]",
        }

        if not visible_rods:
            st.info("표시할 객체가 없습니다.")
        else:
            COLS = 5
            for row_start in range(0, len(visible_rods), COLS):
                batch = visible_rods[row_start: row_start + COLS]
                cols = st.columns(COLS)
                for col, (rod_idx, rod) in zip(cols, batch):
                    df_row = idx_to_row[rod_idx]
                    rid = int(df_row["id"])
                    current_label = df_row["overlap_label"]
                    img_key = st.session_state.image_name

                    thumb = extract_thumbnail(img_bgr, rod)
                    with col:
                        st.image(thumb, caption=f"#{rid}", use_container_width=True)
                        st.markdown(_LABEL_BADGE.get(current_label, ":orange[● 미분류]"))
                        btn_row1 = st.columns(2)
                        btn_row2 = st.columns(2)
                        with btn_row1[0]:
                            if st.button("단일", key=f"s_{rid}_{img_key}", use_container_width=True):
                                clf.add_label(df_row.to_dict(), LABEL_SINGLE)
                                st.session_state.df_measured.loc[
                                    st.session_state.df_measured["id"] == rid, "overlap_label"
                                ] = LABEL_SINGLE
                                st.rerun()
                        with btn_row1[1]:
                            if st.button("겹침", key=f"o_{rid}_{img_key}", use_container_width=True):
                                clf.add_label(df_row.to_dict(), LABEL_OVERLAP)
                                st.session_state.df_measured.loc[
                                    st.session_state.df_measured["id"] == rid, "overlap_label"
                                ] = LABEL_OVERLAP
                                st.rerun()
                        with btn_row2[0]:
                            if st.button("일부", key=f"p_{rid}_{img_key}", use_container_width=True):
                                clf.add_label(df_row.to_dict(), LABEL_PARTIAL)
                                st.session_state.df_measured.loc[
                                    st.session_state.df_measured["id"] == rid, "overlap_label"
                                ] = LABEL_PARTIAL
                                st.rerun()
                        with btn_row2[1]:
                            if st.button("라드 아님", key=f"n_{rid}_{img_key}", use_container_width=True):
                                clf.add_label(df_row.to_dict(), LABEL_NOT_ROD)
                                st.session_state.df_measured.loc[
                                    st.session_state.df_measured["id"] == rid, "overlap_label"
                                ] = LABEL_NOT_ROD
                                st.rerun()

        counts = clf.label_counts()
        st.markdown("---")
        st.markdown(
            f"**누적 학습 데이터**: "
            f"단일 {counts[LABEL_SINGLE]}개 / 겹침 {counts[LABEL_OVERLAP]}개 / "
            f"일부 {counts[LABEL_PARTIAL]}개 / 라드 아님 {counts[LABEL_NOT_ROD]}개  \n"
            f"모델 학습 조건: 최소 2개 클래스 각 5개 이상"
        )


# ════════════════════════════════════════════════════════
# TAB 5 — Manual rod annotation
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ✏️ 수동 라드 표시")
    st.markdown(
        "자동 검출이 실패했을 때 이미지에서 **라드 주위에 사각형을 직접 그려** 측정할 수 있습니다.  \n"
        "표시된 영역은 학습 데이터(단일 라드)로도 자동 저장됩니다."
    )

    try:
        from streamlit_drawable_canvas import st_canvas

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)

        h_orig, w_orig = img_bgr.shape[:2]
        CANVAS_W = min(900, w_orig)
        CANVAS_H = int(h_orig * CANVAS_W / w_orig)
        scale_x = CANVAS_W / w_orig
        scale_y = CANVAS_H / h_orig

        pil_resized = pil_img.resize((CANVAS_W, CANVAS_H), PILImage.LANCZOS)

        st.markdown("**사각형 도구로 라드 영역을 그리세요** (여러 개 가능)")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.15)",
            stroke_width=2,
            stroke_color="#FFD700",
            background_image=pil_resized,
            update_streamlit=True,
            width=CANVAS_W,
            height=CANVAS_H,
            drawing_mode="rect",
            key=f"canvas_{st.session_state.image_name}",
        )

        objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
        n_shapes = len(objects)
        st.caption(f"그려진 사각형: {n_shapes}개")

        # nm/px from scale_info (if available)
        nm_per_px = None
        if scale_info and scale_info.get("success") and scale_info.get("nm_per_pixel"):
            nm_per_px = scale_info["nm_per_pixel"]
            st.caption(f"스케일 정보: {nm_per_px:.4f} nm/px (자동 검출)")
        else:
            nm_per_px_manual = st.number_input(
                "nm / pixel (수동 입력)", min_value=0.001, max_value=10000.0,
                value=1.0, format="%.4f",
                help="스케일바 분석이 안 된 경우 직접 입력하세요. Tab 1에서 설정하면 자동으로 반영됩니다.",
            )
            nm_per_px = nm_per_px_manual

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            process_btn = st.button(
                "라드 처리 및 측정", type="primary",
                disabled=(n_shapes == 0), use_container_width=True
            )
        with col_btn2:
            clear_btn = st.button("수동 라드 초기화", use_container_width=True)

        if clear_btn:
            st.session_state.manual_rods = []
            st.session_state.manual_df = None
            st.rerun()

        if process_btn and objects:
            new_rods = []
            for obj in objects:
                if obj.get("type") != "rect":
                    continue
                # Canvas uses scaleX/scaleY for resize; multiply dimensions
                sx = obj.get("scaleX", 1.0)
                sy = obj.get("scaleY", 1.0)
                rx = int(obj.get("left", 0) / scale_x)
                ry = int(obj.get("top", 0) / scale_y)
                rw = int(obj.get("width", 0) * sx / scale_x)
                rh = int(obj.get("height", 0) * sy / scale_y)

                rx = max(0, min(rx, w_orig - 1))
                ry = max(0, min(ry, h_orig - 1))
                rw = max(1, min(rw, w_orig - rx))
                rh = max(1, min(rh, h_orig - ry))

                feats = _process_manual_rect(img_bgr, rx, ry, rw, rh)
                if feats:
                    new_rods.append(feats)

            st.session_state.manual_rods = new_rods

            if new_rods:
                records = []
                for i, rod in enumerate(new_rods):
                    records.append({
                        "id": f"M{i+1}",
                        "length_nm": round(rod["long_side_px"] * nm_per_px, 2),
                        "diameter_nm": round(rod["short_side_px"] * nm_per_px, 2),
                        "aspect_ratio": round(rod["aspect_ratio"], 3),
                        "center_x": round(rod["center_x"], 1),
                        "center_y": round(rod["center_y"], 1),
                        "area_px2": round(rod["area_px"], 1),
                    })
                    # Add to classifier as "single" training example
                    clf.add_label({
                        "area_px2": rod["area_px"],
                        "aspect_ratio": rod["aspect_ratio"],
                        "solidity": rod["solidity"],
                        "circularity": rod["circularity"],
                        "convexity_defect_count": rod["convexity_defect_count"],
                        "mean_intensity": rod["mean_intensity"],
                        "std_intensity": rod["std_intensity"],
                    }, LABEL_SINGLE)

                st.session_state.manual_df = pd.DataFrame(records)
                st.success(f"{len(new_rods)}개 라드 처리 완료! 학습 데이터에도 추가되었습니다.")
                st.rerun()

        # ── Show results ──────────────────────────────────────────────────────
        if st.session_state.manual_rods:
            st.markdown(f"**처리된 수동 라드: {len(st.session_state.manual_rods)}개**")

            # Annotated preview
            preview = img_bgr.copy()
            for rod in st.session_state.manual_rods:
                box = cv2.boxPoints(rod["rect"])
                box = np.intp(box)
                cv2.drawContours(preview, [box], 0, (0, 215, 255), 2)

            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="수동 표시 라드 (금색 박스)",
                use_container_width=True,
            )

            if st.session_state.manual_df is not None:
                st.dataframe(st.session_state.manual_df, use_container_width=True, hide_index=True)
                csv_m = st.session_state.manual_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "수동 측정 CSV 다운로드", data=csv_m,
                    file_name=f"{Path(st.session_state.image_name).stem}_manual.csv",
                    mime="text/csv",
                )

    except ImportError:
        st.error(
            "`streamlit-drawable-canvas` 패키지가 설치되어 있지 않습니다.\n\n"
            "```\npip install streamlit-drawable-canvas\n```\n\n"
            "설치 후 앱을 재시작하세요."
        )
