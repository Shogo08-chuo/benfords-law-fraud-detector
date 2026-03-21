import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os

# --- 0. 日本語フォント設定（文字化け対策） ---
# ipaexg.ttf が streamlit_app.py と同じフォルダにある前提
FONT_PATH = os.path.join(os.path.dirname(__file__), 'ipaexg.ttf')

if os.path.exists(FONT_PATH):
    jp_font = fm.FontProperties(fname=FONT_PATH)
    # Matplotlibのグローバル設定に反映
    plt.rcParams['font.family'] = jp_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.error(f"フォントファイルが見つかりません。GitHubに ipaexg.ttf をアップロードしてください。パス: {FONT_PATH}")
    jp_font = None

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
            model_name='gemini-2.5-flash',
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except Exception as e:
        st.error(f"モデルの初期化失敗: {e}")
        return None

model = get_model(GENAI_API_KEY)

# --- 2. アプリの基本設定 ---
st.set_page_config(page_title="会計不正検知支援ツール", layout="wide")
st.title("📊 会計データ解析：ベンフォードの法則による異常検知システム")
st.markdown("---")

# --- 3. サイドバー：データ管理 ---
st.sidebar.header("データ管理")
uploaded_file = st.sidebar.file_uploader("解析用CSVアップロード", type="csv")

if st.sidebar.button("🧪 デモ用データを生成"):
    np.random.seed(42)
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    st.session_state['data'] = pd.DataFrame({"amount": data})
    st.sidebar.success("デモデータをロードしました。")

if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 4. メイン解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    
    if "amount" not in df.columns:
        st.error("CSVに 'amount' カラムが必要です。")
        st.stop()

    # データクレンジングと第1桁抽出
    amounts = df["amount"].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    first_digits = amounts.str.lstrip("0").str[0]
    first_digits = first_digits[first_digits.str.isdigit() == True].astype(int)
    
    # 統計計算
    count = len(first_digits)
    observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    benford_dist = np.log10(1 + 1/np.arange(1, 10))
    expected_counts = benford_dist * count
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # 乖離の激しい桁を特定
    diffs = (observed_counts / count) - benford_dist
    max_diff_digit = diffs.idxmax()

    # タブ構成
    tab1, tab2, tab3, tab4 = st.tabs(["📊 解析サマリー", "📈 統計詳細", "🧠 AI調査仮説", "🔎 対象データ確認"])

    with tab1:
        st.subheader("解析サマリー")
        c1, c2, c3 = st.columns(3)
        c1.metric("総データ件数", f"{count:,} 件")
        c2.metric("p値（有意水準 0.05）", f"{p_value:.8f}")
        
        if p_value < 0.05:
            c3.warning("有意な逸脱を確認")
            st.error(f"注意：第1桁「{max_diff_digit}」の出現頻度が理論値から大きく乖離しています。")
        else:
            c3.success("有意な逸脱なし")

        # グラフ描画
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(1, 10), observed_counts / count, alpha=0.4, label='観測分布', color='gray')
        ax.plot(range(1, 10), benford_dist, marker='o', color='red', label='理論分布', linewidth=2)
        
        # フォントが読み込めている場合は日本語を適用
        font_prop = jp_font if jp_font else None
        ax.set_xlabel("第1桁の数字", fontproperties=font_prop)
        ax.set_ylabel("出現確率", fontproperties=font_prop)
        ax.legend(prop=font_prop)
        
        st.pyplot(fig)

    with tab2:
        st.subheader("統計量詳細")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.write("各桁の出現回数と理論値との比較")
            stats_df = pd.DataFrame({
                "観測回数": observed_counts,
                "理論回数": expected_counts.round(1),
                "乖離（％）": (diffs * 100).round(2)
            })
            st.table(stats_df)
        with col_stats2:
            st.write("基本統計量（amount）")
            st.write(df["amount"].describe())

    with tab3:
        st.subheader("AIによる調査仮説の構築")
        if p_value < 0.05 and model:
            prompt = f"""
あなたは熟練した公認会計士です。以下のベンフォード分析結果に基づき専門的な監査レポートを日本語で作成してください。
- 総件数: {count}
- p値: {p_value:.8f}
- 最も乖離が激しい桁: {max_diff_digit}

1.統計的解釈 2.業務上の背景予測 3.リスクシナリオ 4.推奨される詳細調査手続 を出力してください。
"""
            @st.cache_data(show_spinner="AI分析中...")
            def generate_report(_model, _prompt):
                return _model.generate_content(_prompt).text

            try:
                report = generate_report(model, prompt)
                st.markdown(report)
            except Exception as e:
                st.error(f"AI実行エラー: {e}")
        else:
            st.info("有意な差がないためAI分析は不要です。")

    with tab4:
        st.subheader(f"重点調査対象：第1桁が「{max_diff_digit}」のデータ")
        suspicious_data = df[df["amount"].astype(str).str.replace(r'[^0-9.]', '', regex=True).str.lstrip("0").str.startswith(str(max_diff_digit))]
        st.write(f"該当件数: {len(suspicious_data)} 件")
        st.dataframe(suspicious_data, use_container_width=True)

else:
    st.info("サイドバーからCSVをアップロードするか、デモデータを生成してください。")