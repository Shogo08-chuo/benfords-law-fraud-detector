import numpy as np
import pandas as pd
import streamlit as st

from ai_support import BALANCED_MODE, RISK_MODE
from analysis_core import (
    add_default_columns,
    benford_analysis,
    build_hotspots,
    categorize_anomaly,
    enrich_risk_signals,
    generate_demo_dataset,
    load_uploaded_data,
    prepare_amounts,
)
from ui_components import (
    inject_styles,
    render_ai_panel,
    render_hero,
    render_hotspots,
    render_intro_page,
    render_kpi,
    render_limitations_note,
    render_method_notes,
    render_next_actions,
    render_priority_queue,
    render_research_controls,
    render_result_summary,
    render_stage_one_panel,
    render_upload_diagnostics,
    render_workspace_status,
)


st.set_page_config(
    page_title="会計データ確認アプリ",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def init_session_state():
    defaults = {
        "data": None,
        "dataset_label": None,
        "data_source": None,
        "amount_column": None,
        "raw_row_count": None,
        "upload_columns": None,
        "start_time": None,
        "elapsed_time": None,
        "ai_brief": None,
        "ai_brief_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_reference_population_dataset():
    pop = pd.read_csv("city_population.csv")
    pop["transaction_id"] = [f"REF-{i:05d}" for i in range(1, len(pop) + 1)]
    pop["date"] = "2026-04-01"
    pop["department"] = "Reference"
    pop["vendor"] = "Public Statistics"
    pop["employee"] = "System"
    pop["category"] = "Population"
    pop["approval_limit"] = np.nan
    pop["payment_type"] = "N/A"
    pop["region"] = "Reference"
    return pop


def render_data_setup():
    """Keep the primary data controls in this entrypoint to avoid import-cache issues."""
    st.markdown(
        """
        <div class="setup-heading">
            <div>
                <div class="setup-kicker">START HERE</div>
                <div class="setup-title">分析するデータを選ぶ</div>
            </div>
            <div class="setup-copy">サンプルならすぐに試せます。お手元のCSVもここから読み込めます。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    source = st.radio(
        "データの用意方法",
        ["サンプルで試す", "CSVをアップロード"],
        horizontal=True,
        label_visibility="collapsed",
        key="workspace_source",
    )
    controls, ai_control = st.columns([2.2, 1], gap="large")
    with controls:
        if source == "サンプルで試す":
            dataset_name = st.selectbox(
                "サンプルを選ぶ",
                ["購買・経費データ（承認上限寄り）", "参照用: 人口分布に近い自然データ"],
                key="workspace_dataset",
            )
            if st.button("このサンプルで分析する", type="primary", key="load_workspace_sample"):
                st.session_state["data"] = (
                    generate_demo_dataset()
                    if dataset_name == "購買・経費データ（承認上限寄り）"
                    else load_reference_population_dataset()
                )
                st.session_state["dataset_label"] = dataset_name
                st.session_state["data_source"] = "sample"
                st.session_state["amount_column"] = "amount"
                st.session_state["raw_row_count"] = len(st.session_state["data"])
                st.session_state["upload_columns"] = list(st.session_state["data"].columns)
                st.rerun()
        else:
            uploaded_file = st.file_uploader("CSVファイルを選ぶ", type="csv", key="workspace_upload")
            if uploaded_file is not None:
                try:
                    df, detected_col = load_uploaded_data(uploaded_file)
                    st.session_state["data"] = df
                    st.session_state["dataset_label"] = f"アップロードCSV（amount列: {detected_col}）"
                    st.session_state["data_source"] = "upload"
                    st.session_state["amount_column"] = detected_col
                    st.session_state["raw_row_count"] = len(df)
                    st.session_state["upload_columns"] = list(df.columns)
                    st.success("CSVを読み込みました。下の分析結果に反映されます。")
                except Exception as exc:
                    st.error(f"読み込みエラー: {exc}")
    with ai_control:
        st.markdown('<div class="control-label">AIの表示</div>', unsafe_allow_html=True)
        tone_mode = st.radio(
            "AIの表示",
            [BALANCED_MODE, RISK_MODE],
            index=0,
            label_visibility="collapsed",
            key="workspace_ai_tone",
        )
        st.caption("必要なときだけ、AIが確認ポイントを整理します。")
    return tone_mode


def render_analysis_tab(tone_mode):
    if st.session_state["data"] is None:
        render_hero("未選択", tone_mode)
        st.info("画面上部でサンプルまたはCSVを選び、「分析する」を押してください。")
        return

    raw_df = st.session_state["data"].copy()
    prepared_df = prepare_amounts(raw_df)
    if prepared_df.empty:
        st.error("有効な正の金額データが見つかりませんでした。")
        return

    prepared_df = add_default_columns(prepared_df)
    analysis = benford_analysis(prepared_df)
    anomaly_level, anomaly_color = categorize_anomaly(analysis["p_value"], analysis["mad"])
    detailed_df = enrich_risk_signals(prepared_df, analysis["anomaly_digit"], analysis["deviation"])
    vendor_view, department_view, reason_view = build_hotspots(detailed_df)
    valid_count = len(prepared_df)
    raw_count = st.session_state.get("raw_row_count") or len(raw_df)
    excluded_count = max(raw_count - valid_count, 0)

    high_risk_count = int((detailed_df["risk_band"] == "高").sum())
    impacted_vendors = int(vendor_view[vendor_view["high_risk"] > 0]["vendor"].nunique())
    avg_review_score = detailed_df.head(50)["risk_score"].mean()

    render_hero(st.session_state["dataset_label"], tone_mode)
    render_workspace_status(
        dataset_label=st.session_state["dataset_label"],
        data_source=st.session_state.get("data_source"),
        raw_count=raw_count,
        valid_count=valid_count,
        excluded_count=excluded_count,
        amount_column=st.session_state.get("amount_column"),
    )
    if st.session_state.get("data_source") == "upload":
        render_upload_diagnostics(
            raw_count=raw_count,
            valid_count=valid_count,
            excluded_count=excluded_count,
            amount_column=st.session_state.get("amount_column"),
            columns=st.session_state.get("upload_columns") or [],
        )

    render_result_summary(
        anomaly_level=anomaly_level,
        anomaly_color=anomaly_color,
        high_risk_count=high_risk_count,
        impacted_vendors=impacted_vendors,
        anomaly_digit=analysis["anomaly_digit"],
        avg_review_score=avg_review_score,
    )
    render_next_actions(high_risk_count, anomaly_level)
    render_limitations_note()

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi("分析した件数", f"{analysis['total']:,}", "正の金額データだけを対象にしています")
    with kpi_cols[1]:
        render_kpi("全体の違和感", anomaly_level, f"補足指標: p値 {analysis['p_value']:.4f} / MAD {analysis['mad']:.4f}")
    with kpi_cols[2]:
        render_kpi("優先確認件数", f"{high_risk_count:,}", "スコアが高い取引を先に確認します")
    with kpi_cols[3]:
        render_kpi("偏りが目立つベンダー", f"{impacted_vendors:,}", f"上位50件の平均スコア {avg_review_score:.1f}")

    col_left, col_right = st.columns([1.3, 1], gap="large")
    with col_left:
        render_stage_one_panel(analysis, anomaly_level, anomaly_color)
    with col_right:
        render_ai_panel(analysis, detailed_df, vendor_view, department_view, tone_mode)

    render_priority_queue(detailed_df)

    with st.expander("集計と偏りの内訳を見る", expanded=False):
        render_hotspots(vendor_view, department_view, reason_view)
    with st.expander("判定ロジックと手法の説明", expanded=False):
        render_method_notes()
    with st.expander("研究用の補助機能", expanded=False):
        render_research_controls(tone_mode)


def main():
    inject_styles()
    init_session_state()
    tone_mode = render_data_setup()

    st.markdown(
        """
        <div class="app-nav-heading">
            <div>
                <div class="app-kicker">REVIEW WORKSPACE</div>
                <div class="app-name">会計データを確認する</div>
            </div>
            <div class="app-nav-help">データを選択後、<strong>分析結果</strong>を開きます。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("はじめに / 使い方を見る", expanded=st.session_state["data"] is None):
        render_intro_page()

    render_analysis_tab(tone_mode)


if __name__ == "__main__":
    main()
