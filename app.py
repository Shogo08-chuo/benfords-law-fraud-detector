import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. APIの設定（最新の2.5-flashへ更新） ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        # 404エラー対策：2.0から2.5へモデル名を更新
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
st.title("📊 ベンフォードの法則：統計的違和感検知システム")

# --- 3. サイドバー：デモデータ ---
if st.sidebar.button("🧪 デモ用データを生成"):
    np.random.seed(42)
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    st.session_state['data'] = pd.DataFrame({"amount": data})

# --- 4. データ読み込み ---
uploaded_file = st.file_uploader("CSVアップロード", type="csv")
if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 5. 解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    if "amount" in df.columns:
        amounts = df["amount"].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        first_digits = amounts.str.lstrip("0").str[0]
        first_digits = first_digits[first_digits.str.isdigit() == True].astype(int)
        
        observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
        benford_dist = np.log10(1 + 1/np.arange(1, 10))
        expected_counts = benford_dist * observed_counts.sum()
        chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # 結果表示
        st.subheader("1. 統計的解析結果")
        st.metric("p値", f"{p_value:.8f}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, 10), observed_counts / observed_counts.sum(), alpha=0.6, label='観測値')
        ax.plot(range(1, 10), benford_dist, marker='o', color='red', label='理論値')
        st.pyplot(fig)

        # --- 6. AI解説（2.5-flashを使用） ---
        if p_value < 0.05 and model:
            st.divider()
            st.subheader("2. AIによる調査仮説の構築支援")
            
            prompt = f"熟練監査人として、p値 {p_value:.8f} の結果に基づき、1.統計的解釈 2.正当な理由 3.リスクシナリオ 4.推奨アクションを日本語で解説してください。"
            
            @st.cache_data(show_spinner="AIが分析を生成中...")
            def generate_report(_model, _prompt):
                return _model.generate_content(_prompt).text

            try:
                report = generate_report(model, prompt)
                st.markdown(report)
            except Exception as e:
                st.error(f"AI実行エラー: {e}")