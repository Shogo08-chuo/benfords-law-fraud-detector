import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
import os

# --- 1. APIの設定 ---
# 環境変数からAPIキーを取得（GitHub公開時はここを空にするか、os.getenvを使用）
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
else:
    st.error("APIキーが見つかりません。環境変数 'GEMINI_API_KEY' を設定してください。")

# --- 2. アプリの基本設定 ---
st.set_page_config(page_title="会計不正検知ツール", layout="wide")
st.title("ベンフォードの法則による不正検知シミュレーター")
st.write("財務データの『第1桁』の分布を統計的に解析し、不自然なデータ操作を検知します。")

# --- 3. サイドバー：デモデータの生成 ---
st.sidebar.header("設定")
if st.sidebar.button("デモ用データを生成"):
    np.random.seed(42)
    # 正常データ：対数分布に従う数値
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    # 異常データ：特定の数字（5, 8, 1など）から始まる不自然な塊
    fraud_data = np.random.choice([500, 800, 1500, 2500, 5800, 8800], size=300)
    data = np.concatenate([normal_data, fraud_data])
    df_demo = pd.DataFrame({"amount": data})
    st.session_state['data'] = df_demo
    st.sidebar.success("デモデータを読み込みました！")

# --- 4. メイン画面：データの読み込み ---
uploaded_file = st.file_uploader("解析するCSVファイルをアップロードしてください", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df

# --- 5. 解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    st.subheader("1. データプレビュー")
    st.write(df.head())

    if "amount" in df.columns:
        amounts = df["amount"]
        
        # 第1桁を抽出（0や小数点を考慮）
        first_digits = amounts.astype(str).str.replace(".", "", regex=False).str.lstrip("0").str[0]
        first_digits = first_digits.dropna().astype(int)
        
        # 統計計算
        observed_counts = first_digits.value_counts().sort_index()
        digits = np.arange(1, 10)
        observed_counts = observed_counts.reindex(digits, fill_value=0)
        
        # ベンフォードの法則の理論分布
        benford_dist = np.log10(1 + 1/digits)
        expected_counts = benford_dist * observed_counts.sum()
        
        # カイ二乗検定
        chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # --- 解析結果表示 ---
        st.subheader("2. 解析結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("カイ二乗統計量", f"{chi_stat:.2f}")
        col2.metric("p値", f"{p_value:.8f}")
        
        if p_value < 0.05:
            col3.error("異常検知")
            st.warning("このデータはベンフォードの法則から有意に逸脱しています。人為的な操作の可能性があります。")
        else:
            col3.success("正常判定")
            st.info("データは統計的に自然な分布です。")

        # --- グラフ表示 ---
        st.subheader("3. 分布の可視化")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(digits, observed_counts / observed_counts.sum(), alpha=0.6, label='観測値 (実際のデータ)', color='skyblue')
        ax.plot(digits, benford_dist, marker='o', linestyle='--', color='red', label='理論値 (ベンフォードの法則)')
        ax.set_xlabel("第1桁の数字")
        ax.set_ylabel("出現割合")
        ax.set_xticks(digits)
        ax.legend()
        st.pyplot(fig)

        # --- 6. AIによる理解支援 ---
        if p_value < 0.05 and GENAI_API_KEY:
            st.divider()
            st.subheader("4. AIによる理解支援（調査仮説の提示）")
            
            # 最も乖離している桁を特定
            diffs = (observed_counts / observed_counts.sum()) - benford_dist
            max_diff_digit = diffs.idxmax()
            
            prompt = f"""
            あなたは熟練した会計監査人の思考補助アシスタントです。
            ベンフォードの法則を用いた統計解析の結果、以下の異常が検知されました。
            
            【解析結果】
            - p値: {p_value:.8f}（有意な逸脱あり）
            - 最も乖離が大きい桁: {max_diff_digit}
            
            この数値を踏まえ、以下の3点を「断定を避けた表現」で出力してください。
            1. 統計的な歪みが生じる「正当な業務上の理由」の仮説
            2. 潜在的な「不正シナリオ」の仮説
            3. 調査者が次に確認すべき具体的な証憑（領収書、契約書など）やデータ
            """
            
            with st.spinner("AIが分析中..."):
                try:
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    st.error(f"AIの呼び出しでエラーが発生しました: {e}")
        elif p_value < 0.05:
            st.info("※APIキーが設定されていないため、AI解説はスキップされました。")
        
    else:
        st.error("CSVファイルに 'amount' カラムが見つかりません。")