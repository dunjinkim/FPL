"""
SEM Rod Analyzer — Streamlit Web Application

Usage:
    streamlit run app.py
"""

import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from analyzer.scale_bar import detect_scale_bar
from analyzer.rod_detector import detect_rods
from analyzer.measurer import measure_rods, compute_statistics
from analyzer.overlap_classifier import OverlapClassifier, LABEL_SINGLE, LABEL_OVERLAP
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
        "rods": None,           # list[dict]
        "df_measured": None,    # pd.DataFrame (all rods, overlap_label filled)
        "classifier": None,
        "label_pending": {},    # rod_id → label (during this session)
        "analysis_done": False,
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


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 SEM Rod Analyzer")
    st.markdown("---")

    uploaded = st.file_uploader(
        "SEM 이미지 업로드",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="TIF/TIFF, PNG, JPG, BMP 모두 지원합니다.",
    )

    st.markdown("### 분석 파라미터")
    strip_ratio = st.slider(
        "정보 영역 비율 (하단 %)", min_value=5, max_value=30, value=15, step=1,
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

    st.markdown("---")
    st.markdown("### 겹침 분류 모델")
    clf = get_classifier()
    counts = clf.label_counts()
    st.markdown(
        f"학습 데이터: **{counts[LABEL_SINGLE] + counts[LABEL_OVERLAP]}개**  \n"
        f"단일: {counts[LABEL_SINGLE]}개 / 겹침: {counts[LABEL_OVERLAP]}개"
    )
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
        remaining_s = max(0, 5 - counts[LABEL_SINGLE])
        remaining_o = max(0, 5 - counts[LABEL_OVERLAP])
        msg_parts = []
        if remaining_s:
            msg_parts.append(f"단일 {remaining_s}개")
        if remaining_o:
            msg_parts.append(f"겹침 {remaining_o}개")
        if msg_parts:
            st.caption(f"모델 학습까지 라벨 추가 필요: {', '.join(msg_parts)}")

    if st.button("분석 실행", type="primary", disabled=uploaded is None):
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.session_state.image = img_bgr
        st.session_state.image_name = uploaded.name
        st.session_state.analysis_done = False
        st.session_state.label_pending = {}

        with st.spinner("스케일바 검출 중..."):
            scale_info = detect_scale_bar(img_bgr, info_strip_ratio=strip_ratio)
        st.session_state.scale_info = scale_info

        if not scale_info["success"]:
            st.warning(
                "스케일바를 자동으로 검출하지 못했습니다. "
                "'Tab 1 > 스케일바 수동 입력'으로 nm/px 값을 직접 입력해 주세요."
            )

        with st.spinner("라드 검출 중..."):
            rods = detect_rods(
                img_bgr,
                strip_y=scale_info["strip_y"],
                min_area_px=min_area,
                min_aspect_ratio=min_ar,
            )
        st.session_state.rods = rods

        if scale_info["success"] and rods:
            with st.spinner("측정 및 분류 중..."):
                df = measure_rods(rods, scale_info["nm_per_pixel"],
                                  exclude_boundary=exclude_boundary)
                df = clf.classify(df)
            st.session_state.df_measured = df
            st.session_state.analysis_done = True
        elif rods:
            st.info("스케일바 정보를 Tab 1에서 입력 후 '측정 적용' 버튼을 눌러 주세요.")
            # store rods anyway for preview
            st.session_state.df_measured = None
            st.session_state.analysis_done = False


# ── Main area tabs ────────────────────────────────────────────────────────────
if st.session_state.image is None:
    st.markdown(
        """
        ## SEM 실리카 라드 자동 측정 프로그램

        **사용 방법**
        1. 왼쪽 사이드바에서 SEM 이미지를 업로드합니다.
        2. 필요시 분석 파라미터를 조정합니다.
        3. **분석 실행** 버튼을 클릭합니다.
        4. Tab 3에서 측정 결과를 확인하고 CSV/Excel로 다운로드합니다.
        5. Tab 4에서 겹침 라벨링을 진행하면 ML 모델이 자동 분류 정확도를 향상시킵니다.

        **지원 형식**: TIF, TIFF, PNG, JPG, BMP
        """
    )
    st.stop()

img_bgr = st.session_state.image
scale_info = st.session_state.scale_info
rods = st.session_state.rods or []
df_all = st.session_state.df_measured
clf = get_classifier()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📏 스케일바", "🔍 세그멘테이션", "📊 측정 결과", "🏷️ 학습 데이터"]
)

# ════════════════════════════════════════════════════════
# TAB 1 — Scale bar
# ════════════════════════════════════════════════════════
with tab1:
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
            help="스케일바 자동 검출이 안 된 경우, 직접 값을 입력하세요."
        )
        if st.button("측정 적용"):
            if rods:
                df = measure_rods(rods, manual_nm_per_px,
                                  exclude_boundary=exclude_boundary)
                df = clf.classify(df)
                st.session_state.df_measured = df
                st.session_state.analysis_done = True
                # Update scale_info nm_per_pixel
                st.session_state.scale_info["nm_per_pixel"] = manual_nm_per_px
                st.session_state.scale_info["success"] = True
                st.rerun()


# ════════════════════════════════════════════════════════
# TAB 2 — Segmentation
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 세그멘테이션 결과")
    if not rods:
        st.info("분석을 먼저 실행하세요.")
    else:
        col_a, col_b = st.columns(2)

        # Binary image
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        strip_y = scale_info["strip_y"]
        from analyzer.rod_detector import _mask_strip, _binarise, _morphological_clean
        masked = _mask_strip(gray, strip_y)
        binary = _binarise(masked)
        cleaned = _morphological_clean(binary)

        with col_a:
            st.image(cleaned, caption="이진화 이미지 (형태학적 처리 후)", use_container_width=True, clamp=True)

        # Overlay all detected rods
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


# ════════════════════════════════════════════════════════
# TAB 3 — Measurement results
# ════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.analysis_done or df_all is None:
        st.info("분석을 먼저 실행하세요.")
    else:
        df_single = df_all[df_all["overlap_label"] == LABEL_SINGLE].copy()
        df_overlap = df_all[df_all["overlap_label"] == LABEL_OVERLAP].copy()

        st.markdown(
            f"**전체 검출:** {len(df_all)}개 &nbsp;|&nbsp; "
            f"**단일 라드:** {len(df_single)}개 &nbsp;|&nbsp; "
            f"**겹친 라드:** {len(df_overlap)}개"
        )

        # Annotated image
        annotated = annotate_image(img_bgr, rods, df_all, show_measurements=True)
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="초록 = 단일 라드 (측정 포함) / 빨강 = 겹친 라드 (제외)",
            use_container_width=True,
        )

        # ── Statistics ──
        st.markdown("### 통계 요약")
        stats_df = compute_statistics(df_all)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # ── Distribution plots ──
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

        # ── Data table ──
        st.markdown("### 측정 데이터 (단일 라드)")
        display_cols = ["id", "length_nm", "diameter_nm", "aspect_ratio",
                        "center_x", "center_y", "area_px2"]
        if not df_single.empty:
            st.dataframe(df_single[display_cols], use_container_width=True, hide_index=True)

        # ── Downloads ──
        st.markdown("### 다운로드")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            csv_bytes = df_single[display_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV 다운로드", data=csv_bytes,
                file_name=f"{Path(st.session_state.image_name).stem}_rods.csv",
                mime="text/csv",
            )

        with col_dl2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df_single[display_cols].to_excel(writer, sheet_name="Measurements", index=False)
                if not stats_df.empty:
                    stats_df.to_excel(writer, sheet_name="Statistics", index=False)
            excel_buf.seek(0)
            st.download_button(
                "Excel 다운로드", data=excel_buf.read(),
                file_name=f"{Path(st.session_state.image_name).stem}_rods.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col_dl3:
            _, img_encoded = cv2.imencode(".png", annotated)
            st.download_button(
                "주석 이미지 저장", data=img_encoded.tobytes(),
                file_name=f"{Path(st.session_state.image_name).stem}_annotated.png",
                mime="image/png",
            )


# ════════════════════════════════════════════════════════
# TAB 4 — Labelling for ML training
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 학습 데이터 수집")
    st.markdown(
        "각 객체를 보고 **단일 라드**인지 **겹침(클러스터)**인지 라벨을 달아 주세요.  \n"
        "라벨이 쌓이면 사이드바의 **모델 학습** 버튼으로 분류기를 개선할 수 있습니다."
    )

    if not rods or df_all is None:
        st.info("분석을 먼저 실행하세요.")
    else:
        clf = get_classifier()
        idx_to_row = {int(row["_rod_ref"]): row for _, row in df_all.iterrows()}

        # Show objects that are currently in df (not boundary-excluded)
        visible_rods = [(i, rod) for i, rod in enumerate(rods) if i in idx_to_row]

        if not visible_rods:
            st.info("표시할 객체가 없습니다.")
        else:
            COLS = 6
            for row_start in range(0, len(visible_rods), COLS):
                batch = visible_rods[row_start: row_start + COLS]
                cols = st.columns(COLS)
                for col, (rod_idx, rod) in zip(cols, batch):
                    df_row = idx_to_row[rod_idx]
                    rid = int(df_row["id"])
                    current_label = df_row["overlap_label"]

                    thumb = extract_thumbnail(img_bgr, rod)
                    with col:
                        st.image(thumb, caption=f"#{rid}", use_container_width=True)
                        btn_key_s = f"lbl_s_{rid}_{st.session_state.image_name}"
                        btn_key_o = f"lbl_o_{rid}_{st.session_state.image_name}"

                        if current_label == LABEL_SINGLE:
                            st.markdown(":green[● 단일]")
                        elif current_label == LABEL_OVERLAP:
                            st.markdown(":red[● 겹침]")
                        else:
                            st.markdown(":orange[● 미분류]")

                        if st.button("단일", key=btn_key_s, use_container_width=True):
                            clf.add_label(df_row.to_dict(), LABEL_SINGLE)
                            # update in-memory df
                            st.session_state.df_measured.loc[
                                st.session_state.df_measured["id"] == rid, "overlap_label"
                            ] = LABEL_SINGLE
                            st.rerun()

                        if st.button("겹침", key=btn_key_o, use_container_width=True):
                            clf.add_label(df_row.to_dict(), LABEL_OVERLAP)
                            st.session_state.df_measured.loc[
                                st.session_state.df_measured["id"] == rid, "overlap_label"
                            ] = LABEL_OVERLAP
                            st.rerun()

        # Running label summary
        counts = clf.label_counts()
        st.markdown("---")
        st.markdown(
            f"**누적 학습 데이터**: 단일 {counts[LABEL_SINGLE]}개 / 겹침 {counts[LABEL_OVERLAP]}개  \n"
            f"모델 학습에 필요한 최소 데이터: 각 클래스 5개 이상"
        )
