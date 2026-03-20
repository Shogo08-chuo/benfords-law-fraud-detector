import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
import os

# --- 1. APIの設定（無料プラン対応版） ---
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    
    # 利用可能なモデルを動的に取得し、最適なものを選択するロジック
    try:
        # 実際にそのキーで「今」使えるモデルをリストアップ
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 無料枠で通りやすい順に検索（models/ をつけるのがコツ）
        candidates = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
        target_model = next((c for c in candidates if c in available_models), None)
        
        if target_model:
            model = genai.GenerativeModel(target_model)
        else:
            # 最終手段：直接指定（1.5-flashは無料枠の標準です）
            model = genai.GenerativeModel('models/gemini-1.5-flash')
    except Exception as e:
        st.error(f"モデルリストの取得に失敗しました。標準設定で続行します。: {e}")
        model = genai.GenerativeModel('models/gemini-1.5-flash')
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
    # 異常データ：特定の数字から始まる不自然な塊（研究用のシミュレーション）
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
        amounts = df["amount"]
        # 第1桁を抽出
        first_digits = amounts.astype(str).str.replace(".", "", regex=False).str.lstrip("0").str[0]
        first_digits = first_digits.dropna()
        first_digits = first_digits[first_digits.str.isdigit()].astype(int)
        
        # 統計計算
        observed_counts = first_digits.value_counts().sort_index()
        digits = np.arange(1, 10)
        observed_counts = observed_counts.reindex(digits, fill_value=0)
        
        # ベンフォードの法則の理論分布
        benford_dist = np.log10(1 + 1/digits)
        expected_counts = benford_dist * observed_counts.sum()
        
        # カイ二乗検定
        chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # --- 解析結果表示（研究思想に基づき断定を避ける表現へ修正 ） ---
        st.subheader("2. 統計的解析結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("カイ二乗統計量", f"{chi_stat:.2f}")
        col2.metric("p値", f"{p_value:.8f}")
        
        if p_value < 0.05:
            col3.warning("有意な逸脱を確認")
            st.warning("【分析結果】統計的な分布の偏りが確認されました。これは直ちに不正を示すものではなく、背景にある業務プロセスを精査する必要があります。")
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

        # --- 6. AIによる理解支援（ここが研究の核となる部分 [cite: 18, 67]） ---
        if p_value < 0.05 and GENAI_API_KEY:
            st.divider()
            st.subheader("4. 調査仮説の構築支援（LLM）")
            st.markdown("""
            ベンフォード分析を「違和感センサー」として用い[cite: 20]、LLMを違和感の解釈と調査仮説整理に限定して活用します[cite: 21]。
            """)
            
            # 最も乖離している桁を特定
            diffs = (observed_counts / observed_counts.sum()) - benford_dist
            max_diff_digit = diffs.idxmax()
            
            if st.button("AIによる調査仮説の整理を実行"):
                # 不正を断定しない、理解支援のためのプロンプト設計 [cite: 67-72]
                prompt = f"""
あなたは熟練した会計監査人の思考補助アシスタントです。
以下の統計解析結果に基づき、調査者の「理解支援」に特化した情報を整理してください。

【解析結果】
- p値: {p_value:.8f}（有意な逸脱あり）
- 最も乖離が大きい桁: {max_diff_digit}

以下の4項目を「不正を断定しない表現」で出力してください：
1. **統計的解釈**: どの数値にどのような偏りがあるかの要約。
2. **正当な業務上の仮説**: 歪みが生じる妥当なビジネス上の理由。
3. **潜在的なリスクシナリオ**: 注意すべき不正のパターン。
4. **推奨される調査アクション**: 次に確認すべき具体的な証憑や確認項目。
"""
                
                with st.spinner("熟練監査人の視点で分析中..."):
                    try:
                        response = model.generate_content(prompt)
                        st.markdown("### AIによる分析レポート")
                        st.write(response.text)
                        
                        st.caption("---")
                        # 仮説4（H4）の検証用フィードバック [cite: 33, 34]
                        st.radio("このAIの説明は、調査方針の決定に役立ちそうですか？（研究評価用）", 
                                 ["非常に役立つ", "役立つ", "どちらとも言えない", "あまり役立たない"])
                    except Exception as e:
                        st.error(f"AIの呼び出しでエラーが発生しました。APIキーのクォータ（無料枠）を確認してください: {e}")
        elif p_value < 0.05:
            st.info("※APIキーが設定されていないため、AI解説機能は利用できません。")
            
    else:
        st.error("CSVファイルに 'amount' カラムが見つかりません。")