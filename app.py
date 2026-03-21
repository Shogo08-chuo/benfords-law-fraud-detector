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
        # 安定性を考慮し 1.5-flash-latest を使用
        return genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        st.error(f"モデルの初期化失敗: {e}")
        return None

model = get_model(GENAI_API_KEY)

# --- 2. アプリの基本UI設定 ---
st.set_page_config(page_title="会計異常検知ツール", layout="wide")
st.title("📊 会計データ：ベンフォード分布解析システム")
st.markdown("財務データの第1桁の分布を理論値と比較し、統計的な不自然さを可視化します。")
st.markdown("---")

# --- 3. サイドバー：データ管理 & 実験用スイッチ ---
st.sidebar.header("データソース")
uploaded_file = st.sidebar.file_uploader("解析用CSVファイルをアップロード", type="csv")

if st.sidebar.button("🧪 デモ用データをロード"):
    np.random.seed(42)
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    st.session_state['data'] = pd.DataFrame({"amount": data})
    st.sidebar.success("デモデータをロードしました。")

if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 【新規追加】実験用スイッチ (H1-H4検証用) ---
st.sidebar.markdown("---")
st.sidebar.header("研究用設定")
enable_ai = st.sidebar.toggle("AI分析レポートを表示", value=True) # H1/H2/H3用
tone_mode = st.sidebar.radio(
    "AIの回答スタイル",
    ["理解支援モード", "断定モード"],
    index=0,
    help="H4（信頼性評価）の検証用設定です。"
)

# --- 4. メイン解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    if "amount" not in df.columns:
        st.error("CSVファイル内に 'amount' カラムが必要です。")
        st.stop()

    valid_amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    valid_amounts = valid_amounts[valid_amounts > 0]
    first_digits = valid_amounts.astype(str).str.lstrip("0").str[0].astype(int)
    
    total_count = len(first_digits)
    observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    benford_ratios = np.log10(1 + 1/np.arange(1, 10))
    expected_counts = benford_ratios * total_count
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    tab1, tab2, tab3 = st.tabs(["📊 解析結果概要", "🧠 AI分析レポート", "🔎 データエクスプローラー"])

    with tab1:
        st.subheader("分布の解析と可視化")
        col1, col2, col3 = st.columns(3)
        col1.metric("総レコード件数", f"{total_count:,} 件")
        col2.metric("p値 (有意水準 0.05)", f"{p_value:.10f}")
        
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f0f2f6')
        ax.bar(range(1, 10), observed_counts / total_count, alpha=0.5, color='#1f77b4', label='観測値 (実際のデータ)')
        ax.plot(range(1, 10), benford_ratios, marker='s', color='#d62728', label='理論値 (ベンフォードの法則)', linewidth=2)
        
        if jp_font_prop:
            ax.set_xlabel("先頭桁の数字", fontproperties=jp_font_prop)
            ax.set_ylabel("出現確率", fontproperties=jp_font_prop)
            ax.legend(prop=jp_font_prop)
        else:
            ax.set_xlabel("Leading Digit")
            ax.set_ylabel("Frequency Ratio")
            ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("AIによる統計的違和感の分析")
        # 1. AI表示スイッチの判定
        if not enable_ai:
            st.info("研究用設定により、現在AIレポートは非表示です。")
        elif p_value < 0.05 and model:
            diffs = (observed_counts / total_count) - benford_ratios
            max_diff_digit = diffs.idxmax()
            
            # 2. ロールとトーンの固定 (H4検証用)
            if tone_mode == "理解支援モード":
                role_instruction = "あなたは調査者の思考を広げるパートナーです。統計的数値を不正の証拠として断定せず、業務上の自然な理由とリスクの両面から仮説を提示してください。語尾は「〜の可能性があります」「〜という視点が考えられます」としてください。" [cite: 67, 73, 78]
            else:
                role_instruction = "あなたは厳しい会計監査人です。統計的な異常を不正の兆候として強く警告してください。他の可能性を考慮せず、不正のリスクを断定的な口調で指摘してください。語尾は「〜です」「〜と言い切れます」としてください。"

            prompt = f"""
{role_instruction}

【分析データ】
- 総件数: {total_count}
- p値: {p_value:.10f}
- 理論値から最も乖離している桁: {max_diff_digit}

【レポート構成】
1. 統計的結論
2. 乖離の特徴（桁「{max_diff_digit}」の突出）
3. 想定される要因（正当な理由とリスクシナリオの両面）
4. 推奨アクション
"""
            @st.cache_data(show_spinner="AIがデータパターンを分析中...")
            def get_ai_insight(_model, _prompt):
                try:
                    return _model.generate_content(_prompt).text
                except Exception as e:
                    return f"AI呼び出しエラー: {e}"

            report_text = get_ai_insight(model, prompt)
            st.markdown(report_text)
        else:
            st.info("有意な差がないため、AI分析は不要です。")

    with tab3:
        observed_counts_ratio = observed_counts / total_count
        diffs = observed_counts_ratio - benford_ratios
        max_diff_digit = diffs.idxmax()
        st.subheader(f"重点調査対象：先頭桁が「{max_diff_digit}」のデータ")
        valid_amounts_str = valid_amounts.astype(str)
        is_suspicious = valid_amounts_str.str.lstrip("0.").str.startswith(str(max_diff_digit))
        suspicious_df = df.loc[valid_amounts[is_suspicious].index]
        st.write(f"該当件数: {len(suspicious_df)} 件")
        st.dataframe(suspicious_df, use_container_width=True)
        st.subheader("全ソースデータ")
        st.dataframe(df, use_container_width=True)
else:
    st.info("サイドバーからCSVファイルをアップロードするか、デモデータを生成してください。")