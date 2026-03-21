import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os

# --- 0. 日本語フォント設定 ---
FONT_PATH = os.path.join(os.path.dirname(__file__), 'ipaexg.ttf')
if os.path.exists(FONT_PATH):
    jp_font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = jp_font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.error(f"⚠️ フォントファイル(ipaexg.ttf)が見つかりません。")
    jp_font_prop = None

# --- 1. APIの設定 ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name='gemini-2.0-flash', # 実務・実験での速度を考慮し2.0 flashを推奨
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        st.error(f"モデルの初期化失敗: {e}")
        return None

model = get_model(GENAI_API_KEY)

# --- 2. アプリの基本UI設定 ---
st.set_page_config(page_title="Forensic Understanding Support System", layout="wide")
st.title("📊 会計データ：統計的違和感理解支援システム")
st.caption("本システムは不正を断定するものではなく、統計的違和感の解釈を支援する研究用プロトタイプです。")

# --- 3. サイドバー：データ管理 & 研究用設定 ---
with st.sidebar:
    st.header("1. データソース")
    uploaded_file = st.file_uploader("解析用CSVアップロード", type="csv")
    
    if st.button("🧪 デモ用データをロード"):
        np.random.seed(42)
        normal_data = 10 ** np.random.uniform(2, 6, 2000)
        fraud_data = np.random.choice([500, 520, 550, 580], size=300)
        data = np.concatenate([normal_data, fraud_data])
        st.session_state['data'] = pd.DataFrame({"amount": data})
        st.success("デモデータをロードしました。")

    st.divider()
    
    # 第5章 評価方法(H1-H4)のための実験スイッチ [cite: 34, 87]
    st.header("2. 研究用設定 (Research Settings)")
    enable_ai = st.toggle("LLM解説を表示 (H1/H2/H3検証用)", value=True)
    
    # H4: 説明への信頼（断定 vs 支援）の切り替え 
    tone_mode = st.radio(
        "AIの回答スタイル (H4検証用)",
        ["理解支援モード (提案・多角的)", "断定モード (警告・一角的)"],
        index=0,
        help="研究目的（理解支援）に合致するのは『理解支援モード』です。"
    )

if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 4. 解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    if "amount" not in df.columns:
        st.error("CSV内に 'amount' カラムが必要です。")
        st.stop()

    valid_amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    valid_amounts = valid_amounts[valid_amounts > 0]
    first_digits = valid_amounts.astype(str).str.lstrip("0").str[0].astype(int)
    
    total_count = len(first_digits)
    observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    benford_ratios = np.log10(1 + 1/np.arange(1, 10))
    expected_counts = benford_ratios * total_count
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # UI構成
    tab1, tab2, tab3 = st.tabs(["📊 統計解析概要", "🧠 AI分析レポート (解釈支援)", "🔎 データエクスプローラー"])

    with tab1:
        st.subheader("分布の可視化")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("総レコード件数", f"{total_count:,} 件")
            st.metric("p値 (有意差判定)", f"{p_value:.10f}")
            if p_value < 0.05:
                st.warning("⚠️ 統計的に有意な違和感が検知されました。")
            else:
                st.success("✅ 理論分布との顕著な乖離は見られません。")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#f0f2f6')
            ax.bar(range(1, 10), observed_counts / total_count, alpha=0.5, color='#1f77b4', label='観測値')
            ax.plot(range(1, 10), benford_ratios, marker='s', color='#d62728', label='理論値', linewidth=2)
            if jp_font_prop:
                ax.set_xlabel("先頭桁の数字", fontproperties=jp_font_prop)
                ax.set_ylabel("出現確率", fontproperties=jp_font_prop)
                ax.legend(prop=jp_font_prop)
            st.pyplot(fig)

    with tab2:
        st.subheader("AIによる違和感の解釈支援")
        
        if not enable_ai:
            st.info("研究設定によりAI解説は非表示になっています。(LLMなし条件の評価用)")
        elif model:
            diffs = (observed_counts / total_count) - benford_ratios
            max_diff_digit = diffs.idxmax()

            # --- プロンプトの役割固定 (Role Definition) [cite: 67, 78] ---
            if tone_mode == "理解支援モード (提案・多角的)":
                role_prompt = """
あなたは調査者の思考を広げるパートナーです。
統計的な数値を『不正の証拠』として断定するのではなく、なぜその数値が偏っているのか、
業務上の自然な理由（定額支払い、システム上の制約など）と、注意すべきリスクシナリオの両面から
複数の仮説を日本語で提示してください。語尾は「〜の可能性があります」「〜という視点が考えられます」としてください。
                """
            else:
                role_prompt = """
あなたは厳しい会計監査人です。
統計的な異常を『不正の兆候』として強く警告してください。
他の可能性を考慮せず、不正のリスクを断定的な口調で指摘してください。
語尾は「〜です」「〜と言い切れます」としてください。
                """

            prompt = f"""
{role_prompt}

【分析データ】
- p値: {p_value:.10f}
- 最も乖離している桁: {max_diff_digit}

【出力構成】
1. 統計的解釈: p値の意味を分かりやすく。
2. 乖離の背景仮説: 桁「{max_diff_digit}」が突出する「正当な理由」と「リスク」の両面。
3. 推奨アクション: 調査者が次に確認すべき具体的項目。
"""

            @st.cache_data(show_spinner="AIが解釈仮説を生成中...")
            def get_ai_report(_model, _prompt):
                return _model.generate_content(_prompt).text

            report = get_ai_report(model, prompt)
            st.markdown(report)

    with tab3:
        # 異常桁の抽出
        diffs = (observed_counts / total_count) - benford_ratios
        max_digit = diffs.idxmax()
        st.subheader(f"重点確認対象: 先頭桁が「{max_digit}」のデータ")
        is_target = valid_amounts.astype(str).str.lstrip("0.").str.startswith(str(max_digit))
        suspicious_df = df.loc[valid_amounts[is_target].index]
        st.dataframe(suspicious_df, use_container_width=True)

else:
    st.info("データをロードしてください。")