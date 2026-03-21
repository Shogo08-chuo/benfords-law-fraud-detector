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
    jp_font = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = jp_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.error(f"フォントファイル(ipaexg.ttf)がリポジトリに見つかりません。")
    jp_font = None

# --- 1. APIの設定 ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("APIキーが設定されていません。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        return None

model = get_model(GENAI_API_KEY)

# --- 2. UI基本設定 ---
st.set_page_config(page_title="Data Anomaly Detector", layout="wide")
st.title("Financial Data Distribution Analysis")
st.markdown("ベンフォードの法則に基づき、入力データの第1桁分布を解析し、統計的違和感を抽出します。")

# --- 3. サイドバー ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type="csv")

if st.sidebar.button("テスト用データの生成"):
    np.random.seed(42)
    normal = 10 ** np.random.uniform(2, 6, 2000)
    anomaly = np.random.choice([500, 520, 550, 580], size=300) # 5の桁を意図的に増やす
    st.session_state['data'] = pd.DataFrame({"amount": np.concatenate([normal, anomaly])})

if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 4. 解析メイン ---
if 'data' in st.session_state:
    df = st.session_state['data']
    if "amount" in df.columns:
        # データクレンジング
        valid_data = pd.to_numeric(df["amount"], errors='coerce').dropna()
        valid_data = valid_data[valid_data > 0]
        first_digits = valid_data.astype(str).str.lstrip("0").str[0].astype(int)
        
        count = len(first_digits)
        observed = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
        theory_ratios = np.log10(1 + 1/np.arange(1, 10))
        expected = theory_ratios * count
        chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        
        # UI: タブ構成
        t1, t2, t3 = st.tabs(["Analysis Summary", "AI Insight Report", "Raw Data Explorer"])

        with t1:
            st.subheader("Distribution Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", f"{count:,}")
            m2.metric("P-Value", f"{p_value:.10f}")
            
            # グラフ描画
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='#f0f2f6')
            ax.bar(range(1, 10), observed / count, alpha=0.5, color='#1f77b4', label='Observed')
            ax.plot(range(1, 10), theory_ratios, marker='s', color='#d62728', label='Benford Law', linewidth=2)
            
            font_prop = jp_font if jp_font else None
            ax.set_xlabel("Leading Digit", fontproperties=font_prop)
            ax.set_ylabel("Frequency Ratio", fontproperties=font_prop)
            ax.legend(prop=font_prop)
            st.pyplot(fig)

        with t2:
            st.subheader("Statistical Interpretation & Analysis")
            if model:
                # 乖離の激しい桁
                max_diff_digit = ((observed / count) - theory_ratios).idxmax()
                
                # 会計士っぽさを排除した、データ分析的なプロンプト
                prompt = f"""
以下の分析結果に基づき、データサイエンティストの視点から日本語で簡潔にレポートを作成してください。
会計士のような堅苦しい挨拶や形式は一切不要です。

【データ概要】
- 総件数: {count}
- p値: {p_value:.10f}
- 最も理論値から乖離している桁: {max_diff_digit}

【構成項目】
1. 統計的結論: p値に基づき、このデータが自然か不自然かを明快に述べる。
2. 特徴的な偏り: 桁「{max_diff_digit}」が突出していることの意味。
3. 想定される要因: 不正という言葉に限定せず、システム仕様や業務ルール（端数処理、定額設定など）の観点から推測。
4. 推奨される検証項目: データの信頼性を確かめるために、次にどの項目をチェックすべきか。
"""
                @st.cache_data(show_spinner="Analyzing data patterns...")
                def get_ai_insight(_model, _prompt):
                    return _model.generate_content(_prompt).text

                report = get_ai_insight(model, prompt)
                st.markdown(report)
            else:
                st.info("API Key required for AI Insight.")

        with t3:
            # 異常が疑われる桁のデータを最初に見せる
            max_diff_digit = ((observed / count) - theory_ratios).idxmax()
            st.subheader(f"Focus: Records starting with '{max_diff_digit}'")
            filtered_df = df[df["amount"].astype(str).str.lstrip("0").str.startswith(str(max_diff_digit))]
            st.dataframe(filtered_df, use_container_width=True)
            
            st.subheader("All Source Data")
            st.dataframe(df, use_container_width=True)

    else:
        st.error("Column 'amount' not found.")