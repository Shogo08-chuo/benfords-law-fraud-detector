import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. APIの設定（最新モデル・エラー対策版） ---
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    
    try:
        # 2026年現在の推奨モデル 'gemini-2.0-flash' に変更
        # 以前の 'models/gemini-1.5-flash' で発生していた 404/v1beta エラーを回避します
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash', 
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except Exception as e:
        st.error(f"モデルの初期化に失敗しました。: {e}")
else:
    st.error("APIキーが見つかりません。環境変数 'GEMINI_API_KEY' を設定してください。")

# --- 2. アプリの基本設定 ---
st.set_page_config(page_title="会計不正検知支援ツール", layout="wide")
st.title("ベンフォードの法則による統計的違和感検知システム")
st.write("財務データの分布を解析し、統計的な「違和感」を可視化することで、調査者の意思決定を支援します。")

# --- 3. サイドバー：デモデータの生成 ---
st.sidebar.header("設定")
if st.sidebar.button("デモ用データを生成"):
    np.random.seed(42)
    # 正常データ：対数分布に従う数値
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    # 異常データ：特定の数字（5, 8など）から始まる不自然な塊（研究用）
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    df_demo = pd.DataFrame({"amount": data})
    st.session_state['data'] = df_demo
    st.sidebar.success("デモデータを生成しました。")

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
        # 文字列として処理して第1桁を抽出
        amounts = df["amount"].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        first_digits = amounts.str.lstrip("0").str[0]
        first_digits = first_digits[first_digits.str.isdigit() == True].astype(int)
        
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
        st.subheader("2. 統計的解析結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("カイ二乗統計量", f"{chi_stat:.2f}")
        col2.metric("p値", f"{p_value:.8f}")
        
        if p_value < 0.05:
            col3.warning("有意な逸脱を確認")
            st.warning("【分析結果】統計的な分布の偏りが確認されました。業務プロセスを精査する必要があります。")
        else:
            col3.success("有意な逸脱なし")
            st.info("【分析結果】データは統計的に自然な分布の範囲内です。")

        # --- グラフ表示 ---
        st.subheader("3. 分布の可視化")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(digits, observed_counts / observed_counts.sum(), alpha=0.6, label='観測値', color='skyblue')
        ax.plot(digits, benford_dist, marker='o', linestyle='--', color='red', label='理論値')
        ax.set_xlabel("第1桁の数字")
        ax.set_xticks(digits)
        ax.legend()
        st.pyplot(fig)

        # --- 6. AIによる理解支援 ---
        if p_value < 0.05 and GENAI_API_KEY:
            st.divider()
            st.subheader("4. 調査仮説の構築支援（LLM）")
            
            # 最も乖離している桁を特定
            diffs = (observed_counts / observed_counts.sum()) - benford_dist
            max_diff_digit = diffs.idxmax()
            
            if st.button("AIによる調査仮説の整理を実行"):
                prompt = f"""
あなたは熟練した会計監査人の思考補助アシスタントです。
以下の統計解析結果に基づき、調査者の「理解支援」に特化した情報を日本語で整理してください。

【解析結果】
- p値: {p_value:.8f}（有意な逸脱あり）
- 最も乖離が大きい桁: {max_diff_digit}

以下の4項目を出力してください：
1. 統計的解釈: 客観的な要約。
2. 正当な業務上の仮説: 歪みが生じる妥当なビジネス理由。
3. 潜在的なリスクシナリオ: 注意すべき不正のパターン。
4. 推奨される調査アクション: 具体的な証憑や確認項目。
"""
                with st.spinner("熟練監査人の視点で分析中..."):
                    try:
                        # 生成パラメータを追加して安定性を向上
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.GenerationConfig(
                                temperature=0.2, # 安定した回答のため低めに設定
                            )
                        )
                        st.markdown("### AIによる分析レポート")
                        st.write(response.text)
                        
                        st.caption("---")
                        st.radio("このAIの説明は、調査方針の決定に役立ちそうですか？", 
                                 ["非常に役立つ", "役立つ", "どちらとも言えない", "あまり役立たない"])
                    except Exception as e:
                        st.error(f"AIの呼び出しでエラーが発生しました。: {e}")
        elif p_value < 0.05:
            st.info("※APIキーが設定されていないため、AI解説機能は利用できません。")
            
    else:
        st.error("CSVファイルに 'amount' カラムが見つかりません。")