import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. APIの設定（キャッシュ化して無駄な呼び出しを防止） ---
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def get_model(api_key):
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    # 2.0-flashを基本としつつ、エラー対策で安全設定を構築
    return genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

model = get_model(GENAI_API_KEY)

# --- 2. アプリの基本設定 ---
st.set_page_config(page_title="会計不正検知支援ツール", layout="wide")
st.title("ベンフォードの法則による統計的違和感検知システム")

# --- 3. サイドバー：デモデータの生成 ---
if st.sidebar.button("デモ用データを生成"):
    np.random.seed(42)
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    st.session_state['data'] = pd.DataFrame({"amount": data})

# --- 4. メイン画面：データの読み込み ---
uploaded_file = st.file_uploader("解析するCSVファイルをアップロードしてください", type="csv")
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

        # 統計結果表示
        st.subheader("統計的解析結果")
        col1, col2 = st.columns(2)
        col1.metric("p値", f"{p_value:.8f}")
        if p_value < 0.05:
            st.warning("有意な逸脱を確認しました。")
        
        # グラフ表示
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, 10), observed_counts / observed_counts.sum(), alpha=0.6, label='観測値')
        ax.plot(range(1, 10), benford_dist, marker='o', color='red', label='理論値')
        st.pyplot(fig)

        # --- 6. AIによる自動解説（ボタンなし） ---
        if p_value < 0.05 and model:
            st.divider()
            st.subheader("AIによる調査仮説の構築支援")
            
            # 乖離の大きい桁を特定
            max_diff_digit = ((observed_counts / observed_counts.sum()) - benford_dist).idxmax()

            prompt = f"""
あなたは熟練した会計監査人です。p値 {p_value:.8f}、最大乖離桁 {max_diff_digit} という結果に基づき、以下の4点を簡潔に日本語で解説してください。
1.統計的解釈 2.正当な業務上の仮説 3.潜在的なリスクシナリオ 4.推奨される調査アクション
"""
            # AI呼び出しの結果もキャッシュして連打を防ぐ
            @st.cache_data(show_spinner="AI分析を実行中...")
            def generate_ai_report(_model, _prompt):
                try:
                    response = _model.generate_content(_prompt)
                    return response.text
                except Exception as e:
                    return f"AI呼び出しエラー: {e}"

            report = generate_ai_report(model, prompt)
            st.markdown(report)