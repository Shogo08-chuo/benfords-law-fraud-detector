from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st

from ai_support import create_ai_prompt, get_ai_insight, get_gas_url
from analysis_core import high_risk_signature


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;800&display=swap');
        :root {
            --bg: #ffffff;
            --surface: #ffffff;
            --line: #d9e0e7;
            --text: #111827;
            --muted: #4b5563;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            font-family: "Noto Sans JP", sans-serif;
            background: #ffffff !important;
            color: #111827 !important;
        }
        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }
        .stAppToolbar { display: none; }
        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2.4rem;
            max-width: 1320px;
        }
        h1, h2, h3, p, li, label, strong {
            font-family: "Noto Sans JP", sans-serif !important;
        }
        span[class*="material-symbols"],
        i[class*="material-symbols"] {
            font-family: "Material Symbols Rounded" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased;
        }
        .hero-card, .panel-card, .kpi-card, .info-card, .sidebar-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: 0 2px 8px rgba(17, 24, 39, 0.06);
            backdrop-filter: none;
        }
        .hero-card {
            padding: 1.8rem 2rem;
            margin-bottom: 1.1rem;
            position: relative;
            overflow: hidden;
        }
        .hero-card::before {
            content: "";
            position: absolute;
            inset: -20% auto auto 55%;
            width: 320px;
            height: 320px;
            background: none;
            pointer-events: none;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.85fr);
            gap: 1rem;
            align-items: end;
            position: relative;
            z-index: 1;
        }
        .hero-eyebrow {
            color: #7dd3fc;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            color: #111827;
            line-height: 1.05;
            margin-top: 0.6rem;
            max-width: 760px;
        }
        .hero-copy {
            color: #4b5563;
            font-size: 1rem;
            margin-top: 0.9rem;
            max-width: 720px;
            line-height: 1.8;
        }
        .hero-status {
            background: #f8fafc;
            border: 1px solid #d9e0e7;
            border-radius: 22px;
            padding: 1rem 1.1rem;
        }
        .hero-status-label {
            color: #8ca3c7;
            font-size: 0.8rem;
            margin-bottom: 0.2rem;
        }
        .hero-status-value {
            color: #111827;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.45;
        }
        .story-band {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.8rem 0 1.2rem 0;
        }
        .story-card, .signal-item {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem;
        }
        .status-strip, .summary-card, .notice-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .status-label, .summary-label {
            color: #637689;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .status-value, .summary-value {
            color: #17324d;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.25rem;
            line-height: 1.4;
        }
        .status-note, .summary-note {
            color: #617286;
            font-size: 0.84rem;
            line-height: 1.55;
            margin-top: 0.25rem;
        }
        .summary-card {
            border-color: #cfe2e5;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfb 100%);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: 1.35fr 1fr 1fr;
            gap: 0.9rem;
        }
        .summary-lead {
            color: #17324d;
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.45;
            margin-top: 0.35rem;
        }
        .summary-metric {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.95rem;
        }
        .notice-card {
            background: #eff8f7;
            border-color: #cfeae5;
        }
        .notice-title {
            color: #17324d;
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .notice-copy {
            color: #55736f;
            font-size: 0.86rem;
            line-height: 1.65;
        }
        .action-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.9rem 0 1.1rem;
        }
        .action-card {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem;
        }
        .action-title {
            color: #17324d;
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .action-copy {
            color: #617286;
            font-size: 0.86rem;
            line-height: 1.6;
        }
        .story-step {
            color: #7dd3fc;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .story-title, .info-title {
            color: #111827;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .story-copy, .info-copy {
            color: #4b5563;
            font-size: 0.9rem;
            line-height: 1.65;
        }
        .kpi-card {
            padding: 1rem 1.15rem;
            min-height: 144px;
        }
        .kpi-label {
            color: #8ca3c7;
            font-size: 0.84rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
            letter-spacing: 0.06em;
        }
        .kpi-value {
            color: #111827;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.45rem;
        }
        .kpi-note, .muted {
            color: #4b5563;
            font-size: 0.9rem;
            line-height: 1.7;
        }
        .panel-card {
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .panel-title, .section-header {
            color: #111827;
            font-weight: 800;
        }
        .panel-title { font-size: 1.12rem; margin-bottom: 0.2rem; }
        .panel-subtitle {
            color: #4b5563;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            line-height: 1.65;
        }
        .stage-pill {
            display: inline-block;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.12);
            color: #7dd3fc;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            border: 1px solid rgba(56, 189, 248, 0.18);
        }
        .section-header {
            font-size: 1.2rem;
            margin: 0.45rem 0 0.2rem 0;
        }
        .signal-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 0.5rem;
        }
        .sidebar-card {
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 1rem;
            border-radius: 22px;
        }
        .sidebar-title {
            color: #f8fbff;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sidebar-copy {
            color: #9eb0c8;
            font-size: 0.84rem;
            line-height: 1.6;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d9e0e7;
        }
        [data-testid="stSidebar"] * { color: #dbe6f5; }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stFileUploader > label {
            color: #9eb0c8 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.04em;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            overflow: hidden;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(56, 189, 248, 0.22);
            background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
            color: white;
            font-weight: 700;
            min-height: 2.8rem;
            padding: 0.55rem 1rem;
            box-shadow: 0 14px 32px rgba(8, 145, 178, 0.22);
        }
        .stExpander {
            border: 1px solid rgba(148, 163, 184, 0.14) !important;
            border-radius: 20px !important;
            background: #ffffff !important;
        }
        /* Light, high-contrast workspace theme */
        :root {
            --bg: #f4f7fa;
            --surface: #ffffff;
            --line: #dce5ed;
            --text: #16263a;
            --muted: #617286;
        }
        .stApp {
            background: linear-gradient(180deg, #f9fbfd 0%, #f2f6f9 100%);
            color: var(--text);
        }
        .block-container { padding-top: 1.4rem; }
        .hero-card, .panel-card, .kpi-card, .info-card, .sidebar-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 8px 28px rgba(22, 38, 58, 0.06);
            backdrop-filter: none;
        }
        .app-nav-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            margin: 0 0 0.85rem;
        }
        .app-kicker, .hero-eyebrow {
            color: #087d78;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
        }
        .app-name { color: #17324d; font-size: 1.35rem; font-weight: 800; margin-top: 0.15rem; }
        .app-nav-help { color: var(--muted); font-size: 0.86rem; }
        .app-nav-help strong { color: #087d78; }
        .setup-heading {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1.5rem;
            background: #ffffff;
            border: 1px solid #dce5ed;
            border-bottom: 0;
            border-radius: 18px 18px 0 0;
            padding: 1.2rem 1.35rem 0.9rem;
        }
        .setup-kicker { color: #087d78; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.12em; }
        .setup-title { color: #17324d; font-size: 1.15rem; font-weight: 800; margin-top: 0.2rem; }
        .setup-copy { color: #617286; font-size: 0.84rem; }
        .control-label { color: #17324d; font-size: 0.82rem; font-weight: 800; margin: 0.2rem 0 0.55rem; }
        div[data-testid="stRadio"]:has(input[name="workspace_source"]) {
            background: #ffffff;
            border: 1px solid #dce5ed;
            border-top: 0;
            border-bottom: 0;
            padding: 0 1.35rem 0.9rem;
            margin: 0;
        }
        div[data-testid="stRadio"]:has(input[name="workspace_source"]) > div { gap: 0.4rem; }
        div[data-testid="stRadio"]:has(input[name="workspace_source"]) label {
            background: #f3f7f8;
            border: 1px solid #dce5ed;
            border-radius: 999px;
            color: #52667a !important;
            font-size: 0.85rem;
            font-weight: 700;
            margin: 0 !important;
            padding: 0.38rem 0.75rem;
        }
        div[data-testid="stRadio"]:has(input[name="workspace_source"]) label:has(input:checked) {
            background: #e6f5f2;
            border-color: #9ed8d0;
            color: #087d78 !important;
        }
        div[data-testid="stRadio"]:has(input[name="workspace_source"]) input { display: none; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(input[name="workspace_source"]) {
            background: #fff;
            border: 1px solid #dce5ed;
            border-top: 0;
            border-radius: 0 0 18px 18px;
            box-shadow: 0 8px 28px rgba(22, 38, 58, 0.06);
            margin-bottom: 1.6rem;
            padding: 0 1.35rem 1.1rem;
        }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #f4f7fa !important; }
        .hero-card { border-color: #cfe2e5; }
        .hero-card::before { background: radial-gradient(circle, rgba(36, 161, 151, 0.16), transparent 65%); }
        .hero-title, .hero-status-value, .story-title, .info-title, .kpi-value, .panel-title, .section-header, .sidebar-title { color: #17324d; }
        .hero-copy, .story-copy, .info-copy, .kpi-note, .muted, .panel-subtitle, .sidebar-copy { color: #617286; }
        .hero-status { background: #eff8f7; border-color: #cfeae5; border-radius: 16px; }
        .hero-status-label, .kpi-label { color: #637689; }
        .story-card, .signal-item { background: #fff; border-color: var(--line); border-radius: 16px; }
        .story-step { color: #087d78; }
        .stage-pill { background: #e6f5f2; color: #087d78; border-color: #bde5df; }
        .info-card { padding: 1.2rem; min-height: 160px; }
        .signal-item strong { display: block; color: #17324d; margin-bottom: 0.35rem; }
        .signal-item span { color: #617286; font-size: 0.88rem; line-height: 1.6; }
        [data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"] { background: #fff !important; }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stFileUploader > label { color: var(--muted) !important; }
        [data-testid="stDataFrame"] { border-color: var(--line); border-radius: 14px; }
        .stButton > button { border-color: #087d78; background: #087d78; box-shadow: 0 6px 14px rgba(8, 125, 120, 0.18); }
        .stExpander { border-color: var(--line) !important; border-radius: 14px !important; background: #fff !important; }
        .stAlert { border-radius: 14px; }
        [data-testid="stSidebar"] .stButton > button { width: 100%; }
        .sidebar-section-label {
            color: #17324d;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin: 1.35rem 0 0.55rem;
        }
        .sidebar-guide {
            background: #eff8f7;
            border: 1px solid #cfeae5;
            border-radius: 12px;
            color: #55736f !important;
            font-size: 0.82rem;
            line-height: 1.6;
            margin-top: 1.4rem;
            padding: 0.85rem;
        }
        .sidebar-guide strong { color: #087d78 !important; }
        @media (max-width: 700px) {
            .app-nav-heading { align-items: start; flex-direction: column; gap: 0.35rem; }
            .setup-heading { align-items: start; flex-direction: column; gap: 0.35rem; }
            .hero-title { font-size: 1.8rem; }
            .hero-grid, .story-band, .signal-list, .status-strip, .summary-grid, .action-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero(dataset_label, tone_mode):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-eyebrow">会計データの確認</div>
                    <div class="hero-title">会計データの確認アプリ</div>
                    <div class="hero-copy">
                        数字の偏りを見て、確認した方がよい取引を上から表示します。
                    </div>
                </div>
                <div class="hero-status">
                    <div class="hero-status-label">現在の表示</div>
                    <div class="hero-status-value">データ: {dataset_label}</div>
                    <div class="hero-status-value">AIの表示: {tone_mode}</div>
                    <div class="hero-status-label" style="margin-top:0.7rem;">使い方</div>
                    <div class="hero-status-value">数字を見る → 優先確認リストを見る → 必要ならAIで整理する</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_intro_page():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-eyebrow">概要</div>
                    <div class="hero-title">このアプリでできること</div>
                    <div class="hero-copy">
                        会計データの数字の偏りを見て、優先して確認した方がよい取引を見つけるためのアプリです。
                    </div>
                </div>
                <div class="hero-status">
                    <div class="hero-status-label">このアプリがすること</div>
                    <div class="hero-status-value">数字の偏りを確認する</div>
                    <div class="hero-status-value">確認順を出す</div>
                    <div class="hero-status-value">確認観点を整理する</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">ベンフォードの法則とは？</div>
                <div class="info-copy">
                    自然に集まった数字では、先頭の数字が均等には出ません。<br><br>
                    たとえば <strong>1</strong> は多く、<strong>9</strong> は少なく出る傾向があります。<br><br>
                    この傾向から大きく外れているときは、入力ミス、承認ルールの影響、人為的な調整などを確認するきっかけになります。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card" style="margin-top:1rem;">
                <div class="info-title">このアプリの見方</div>
                <div class="info-copy">
                    1. まず数字の偏りチェックを見ます。<br>
                    2. 次に優先確認リストを見ます。<br>
                    3. 必要ならAI分析を実行して、確認ポイントを整理します。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">こういうときに使う</div>
                <div class="info-copy">
                    経費精算の一覧をざっと確認したいとき。<br>
                    全件を見る時間がないとき。<br>
                    どこから確認を始めるべきか迷うとき。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="info-card" style="margin-top:1rem;">
                <div class="info-title">注意点</div>
                <div class="info-copy">
                    偏りがある = 不正、ではありません。<br>
                    データ件数が少ないと参考になりにくいです。<br>
                    上限金額や定額契約があると、自然に偏ることがあります。<br>
                    AIの出力は最終判断ではなく、確認の補助です。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">最初に試す流れ</div>
            <div class="panel-subtitle">まずはサンプルデータで全体の流れをつかめます。</div>
            <div class="signal-list">
                <div class="signal-item">
                    <strong>1. データを読み込む</strong>
                    <span>画面上部からサンプルデータを選び、読み込みます。</span>
                </div>
                <div class="signal-item">
                    <strong>2. 数字の偏りを見る</strong>
                    <span>異常度とグラフを見て、一覧全体に違和感があるかを確認します。</span>
                </div>
                <div class="signal-item">
                    <strong>3. 上位の取引を見る</strong>
                    <span>優先確認リストの上から順に、確認した方がよい取引を見ていきます。</span>
                </div>
                <div class="signal-item">
                    <strong>4. 必要ならAIで整理する</strong>
                    <span>確認観点を短く整理したいときだけAI分析を実行します。</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_status(dataset_label, data_source, raw_count, valid_count, excluded_count, amount_column):
    source_label = "アップロードCSV" if data_source == "upload" else "サンプルデータ"
    st.markdown(
        f"""
        <div class="status-strip">
            <div>
                <div class="status-label">データ</div>
                <div class="status-value">{dataset_label}</div>
                <div class="status-note">現在の分析対象です</div>
            </div>
            <div>
                <div class="status-label">読み込み元</div>
                <div class="status-value">{source_label}</div>
                <div class="status-note">金額列: {amount_column or "amount"}</div>
            </div>
            <div>
                <div class="status-label">元データ件数</div>
                <div class="status-value">{raw_count:,} 件</div>
                <div class="status-note">読み込み時点の総件数</div>
            </div>
            <div>
                <div class="status-label">分析対象件数</div>
                <div class="status-value">{valid_count:,} 件</div>
                <div class="status-note">除外 {excluded_count:,} 件</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_diagnostics(raw_count, valid_count, excluded_count, amount_column, columns):
    shown_columns = " / ".join(columns[:8]) if columns else "-"
    extra = max(len(columns) - 8, 0)
    suffix = f" ほか {extra} 列" if extra else ""
    st.markdown(
        f"""
        <div class="notice-card">
            <div class="notice-title">CSV読み込み結果</div>
            <div class="notice-copy">
                金額列として <strong>{amount_column}</strong> を使用しています。<br>
                読み込み {raw_count:,} 件のうち、分析に使ったのは {valid_count:,} 件です。<br>
                金額が空欄・数値以外・0以下の行は {excluded_count:,} 件除外しています。<br>
                列: {shown_columns}{suffix}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_summary(anomaly_level, anomaly_color, high_risk_count, impacted_vendors, anomaly_digit, avg_review_score):
    if anomaly_level == "高":
        lead = "全体として数字の偏りが強く、優先確認対象が明確にあります。"
    elif anomaly_level == "中":
        lead = "一部に偏りがあり、上位の取引から順に確認するのが妥当です。"
    else:
        lead = "全体の偏りは強くありません。局所的な違和感がないか上位だけ確認します。"

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-grid">
                <div>
                    <div class="summary-label">結論</div>
                    <div class="summary-lead">{lead}</div>
                    <div class="summary-note">この結果は不正の断定ではなく、確認順を決めるための補助です。</div>
                </div>
                <div class="summary-metric">
                    <div class="summary-label">主な理由</div>
                    <div class="summary-value" style="color:{anomaly_color};">異常度 {anomaly_level}</div>
                    <div class="summary-note">先頭数字の <strong>{anomaly_digit}</strong> に偏りが出ています。</div>
                </div>
                <div class="summary-metric">
                    <div class="summary-label">まず見る範囲</div>
                    <div class="summary-value">{high_risk_count:,} 件 / {impacted_vendors:,} ベンダー</div>
                    <div class="summary-note">上位50件の平均スコアは {avg_review_score:.1f} です。</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_next_actions(high_risk_count, anomaly_level):
    third = "必要ならAIで確認観点を整理します。" if high_risk_count > 0 else "必要なら集計から偏りの理由を見ます。"
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">次にやること</div>
            <div class="panel-subtitle">迷わないように、確認の順番を固定しています。</div>
            <div class="action-grid">
                <div class="action-card">
                    <div class="action-title">1. 結論を確認する</div>
                    <div class="action-copy">全体の違和感は <strong>{anomaly_level}</strong> です。まずは上の要約だけ把握します。</div>
                </div>
                <div class="action-card">
                    <div class="action-title">2. 優先確認リストを見る</div>
                    <div class="action-copy">高リスク {high_risk_count:,} 件を上から見れば、確認開始位置で迷いません。</div>
                </div>
                <div class="action-card">
                    <div class="action-title">3. 理由を深掘りする</div>
                    <div class="action-copy">{third}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_limitations_note():
    st.markdown(
        """
        <div class="notice-card" style="margin-top:-0.2rem;">
            <div class="notice-title">この結果の扱い</div>
            <div class="notice-copy">
                偏りがあること自体は、不正の証明ではありません。承認上限、定額契約、価格設定、月末処理などの業務要因でも偏りは出ます。
                この画面は、全件確認の代わりに「どこから見るか」を決めるために使います。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_benford_chart(analysis):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(1, 10)),
            y=analysis["actual_ratio"],
            name="実際の分布",
            marker_color="#0f766e",
            opacity=0.78,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 10)),
            y=analysis["expected_ratio"],
            mode="lines+markers",
            name="理論上の分布",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="先頭の数字",
        yaxis_title="割合",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def render_ai_panel(analysis, detailed_df, vendor_view, department_view, tone_mode):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-pill">AIの整理</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">AIによる確認ポイント整理</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtitle">必要なときだけAIで確認ポイントを整理します。</div>',
        unsafe_allow_html=True,
    )

    prompt = create_ai_prompt(analysis, detailed_df, vendor_view, department_view)
    brief_key = f"{tone_mode}:{analysis['total']}:{analysis['anomaly_digit']}:{high_risk_signature(detailed_df)}"

    run_ai = st.button("AI分析を実行", use_container_width=True)
    if run_ai:
        insight, error = get_ai_insight(prompt, tone_mode)
        st.session_state["ai_brief"] = insight
        st.session_state["ai_brief_key"] = brief_key
        if error:
            st.session_state["ai_brief"] = None
            st.info(error)

    if st.session_state.get("ai_brief_key") == brief_key and st.session_state.get("ai_brief"):
        st.markdown(st.session_state["ai_brief"])
    else:
        st.markdown(
            """
            **確認観点**
            - 正当理由候補: 承認上限、定額契約、価格設定ルール、定常発注
            - リスク候補: 上限回避、同額多発、月末集中、特定ベンダー偏重
            - 次の確認: 証憑、承認者、発注根拠、ベンダー重複、月末処理ルール
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_stage_one_panel(analysis, anomaly_level, anomaly_color):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="stage-pill">数字の確認</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">数字の偏りチェック</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtitle">まず全体傾向を見ます。統計値は補足として下に置いています。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="muted">
        異常度: <strong style="color:{anomaly_color};">{anomaly_level}</strong><br>
        ずれが大きい数字: <strong>{analysis["anomaly_digit"]}</strong><br>
        一番大きい確認理由: <strong>先頭数字の偏り</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(render_benford_chart(analysis), use_container_width=True)
    with st.expander("統計値の詳細を見る", expanded=False):
        st.markdown(
            f"""
            - p値: **{analysis["p_value"]:.6f}**
            - χ²: **{analysis["chi_stat"]:.3f}**
            - MAD: **{analysis["mad"]:.4f}**
            - ずれが最も大きい先頭数字: **{analysis["anomaly_digit"]}**
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_priority_queue(detailed_df):
    st.markdown('<div class="section-header">優先確認リスト</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">上から順に見れば十分です。まずは高リスクだけに絞る前提で作っています。</div>',
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([1, 1, 1.2])
    with filter_cols[0]:
        min_risk = st.slider("最低リスクスコア", 0, 100, 70, 5)
    with filter_cols[1]:
        bands = st.multiselect("異常度", ["高", "中", "低"], default=["高"])
    with filter_cols[2]:
        departments = st.multiselect("部門", sorted(detailed_df["department"].dropna().unique()), default=[])

    filtered = detailed_df[detailed_df["risk_score"] >= min_risk]
    if bands:
        filtered = filtered[filtered["risk_band"].astype(str).isin(bands)]
    if departments:
        filtered = filtered[filtered["department"].isin(departments)]

    st.caption(f"表示件数: {len(filtered):,} / {len(detailed_df):,} 件")

    review_columns = [
        "transaction_id",
        "date",
        "department",
        "vendor",
        "employee",
        "category",
        "amount",
        "approval_limit",
        "risk_score",
        "risk_band",
        "risk_reasons",
    ]
    st.dataframe(
        filtered[review_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "amount": st.column_config.NumberColumn(format="¥%.0f"),
            "approval_limit": st.column_config.NumberColumn(format="¥%.0f"),
            "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100),
        },
    )

    if not filtered.empty:
        top_case = filtered.iloc[0]
        st.markdown(
            f"""
            **最優先確認候補**
            ` {top_case['transaction_id']} ` / {top_case['vendor']} / {top_case['department']} / ¥{top_case['amount']:,.0f}
            / スコア {top_case['risk_score']} / {top_case['risk_reasons']}
            """
        )
    else:
        st.info("条件に合う取引はありません。最低リスクスコアか絞り込み条件を調整してください。")


def render_hotspots(vendor_view, department_view, reason_view):
    st.markdown('<div class="section-header">集計</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">ベンダーや部門ごとの偏りをまとめて見ます。</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ベンダー別**")
        st.dataframe(
            vendor_view.head(8),
            use_container_width=True,
            hide_index=True,
            column_config={
                "total_amount": st.column_config.NumberColumn(format="¥%.0f"),
                "avg_risk": st.column_config.NumberColumn(format="%.1f"),
            },
        )
    with col2:
        st.markdown("**部門別**")
        st.dataframe(
            department_view.head(8),
            use_container_width=True,
            hide_index=True,
            column_config={
                "total_amount": st.column_config.NumberColumn(format="¥%.0f"),
                "avg_risk": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    reason_fig = go.Figure(
        go.Bar(
            x=reason_view["transactions"],
            y=reason_view["primary_reason"],
            orientation="h",
            marker_color="#2563eb",
            text=reason_view["transactions"],
            textposition="auto",
        )
    )
    reason_fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="件数",
        yaxis_title="主な理由",
    )
    st.plotly_chart(reason_fig, use_container_width=True)


def render_method_notes():
    st.markdown(
        """
        - まず数字の偏りを確認します。
        - そのあとAIで確認ポイントを整理します。
        - ベンフォード分析はすべてのデータに有効ではありません。価格設定ルールや承認上限の影響を強く受ける場合があります。
        - 本アプリでは、キリの良い金額、承認上限直前、同額多発、月末集中などの補助シグナルも併用しています。
        """
    )


def render_research_controls(tone_mode):
    st.markdown("補助機能です。通常は閉じたままで問題ありません。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 調査開始"):
            st.session_state["start_time"] = datetime.now().timestamp()
            st.session_state["elapsed_time"] = None
    with col2:
        if st.button("⏹️ 調査終了"):
            if st.session_state["start_time"]:
                st.session_state["elapsed_time"] = datetime.now().timestamp() - st.session_state["start_time"]
                st.session_state["start_time"] = None

    if st.session_state["start_time"]:
        st.warning("調査計測中")
    elif st.session_state["elapsed_time"]:
        st.success(f"所要時間: {st.session_state['elapsed_time']:.1f} 秒")

    with st.form("evaluation_form"):
        q1 = st.slider("H1: 着目箇所の明確化", 1, 5, 3)
        q2 = st.slider("H2: 仮説の多様性", 1, 5, 3)
        q4 = st.slider("H4: 理解支援としての信頼性", 1, 5, 3)
        submitted = st.form_submit_button("評価データを記録")

        if submitted:
            gas_url = get_gas_url()
            if not gas_url:
                st.error("GAS_URL が未設定です。")
            else:
                elapsed = st.session_state.get("elapsed_time")
                payload = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "style": tone_mode,
                    "q1": str(q1),
                    "q2": str(q2),
                    "time": str(round(elapsed, 1) if elapsed else 0),
                    "q4": str(q4),
                }
                try:
                    response = requests.post(gas_url, data=payload, timeout=15)
                    if response.status_code == 200 and "Success" in response.text:
                        st.success("評価データを記録しました。")
                    else:
                        st.error(f"送信失敗: {response.status_code}")
                        st.caption(response.text[:300])
                except Exception as exc:
                    st.error(f"通信エラー: {exc}")
