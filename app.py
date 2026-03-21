import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. APIの設定（有料枠・キャッシュ最適化版） ---
# Streamlit Cloudの「Secrets」に設定したキーを読み込みます
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    """モデルの初期化を1回だけに限定し、有料枠の効率を高めます"""
    try:
        genai.configure(api_key=api_key)
        # 課金プランでは 2.0-flash がコスパ・速度ともに最強です
        return genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except Exception as e:
        st.error(f"モデルの初期化に失敗しました: {e}")
        return None

model = get_model(GENAI_API_KEY)

# --- 2. アプリの基本設定 ---
st.set_page_config(page_title="会計不正検知支援ツール", layout="wide")
st.title("📊 ベンフォードの法則：統計的違和感検知システム")
st.write("財務データの分布を解析し、統計的な「違和感」をAIと共に深掘りします。")

# --- 3. サイドバー：デモデータの生成 ---
st.sidebar.header("データ設定")
if st.sidebar.button("🧪 デモ用データを生成"):
    np.random.seed(42)
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    st.session_state['data'] = pd.DataFrame({"amount": data})
    st.sidebar.success("デモデータをロードしました。")

# --- 4. メイン画面：データの読み込み ---
uploaded_file = st.file_uploader("解析するCSVファイルをアップロードしてください", type="csv")
if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 5. 解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    
    if "amount" in df.columns:
        # データクレンジングと第1桁抽出
        amounts = df["amount"].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        first_digits = amounts.str.lstrip("0").str[0]
        first_digits = first_digits[first_digits.str.isdigit() == True].astype(int)
        
        # 統計計算
        observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
        benford_dist = np.log10(1 + 1/np.arange(1, 10))
        expected_counts = benford_dist * observed_counts.sum()
        chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # --- 解析結果表示 ---
        st.subheader("1. 統計的解析結果")
        col1, col2, col3 = st.columns(3)
        col1.metric("カイ二乗統計量", f"{chi_stat:.2f}")
        col2.metric("p値", f"{p_value:.8f}")
        
        if p_value < 0.05:
            col3.warning("⚠️ 有意な逸脱あり")
            st.warning("統計的な分布の偏りがあります。AIによる要因分析を確認してください。")
        else:
            col3.success("✅ 有意な逸脱なし")

        # グラフ表示
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, 10), observed_counts / observed_counts.sum(), alpha=0.6, label='観測値', color='skyblue')
        ax.plot(range(1, 10), benford_dist, marker='o', linestyle='--', color='red', label='理論値(ベンフォード)')
        ax.set_xlabel("第1桁の数字")
        ax.set_ylabel("出現確率")
        ax.legend()
        st.pyplot(fig)

        # --- 6. AIによる自動解説（有料枠・キャッシュ保護） ---
        if p_value < 0.05 and model:
            st.divider()
            st.subheader("2. AIによる調査仮説の構築支援")
            
            # 乖離の大きい桁を特定
            diffs = (observed_counts / observed_counts.sum()) - benford_dist
            max_diff_digit = diffs.idxmax()

            prompt = f"""
あなたは熟練した会計監査人の思考補助アシスタントです。
統計解析の結果、p値が {p_value:.8f} となり、特に「{max_diff_digit}」から始まる数値に強い偏りが見られました。

以下の4項目について、調査者の理解を支援するレポートを作成してください。
1. 統計的解釈: どのような偏りがあるか客観的な要約。
2. 正当な業務上の仮説: 不正ではなく、業務特性（単価設定など）で説明がつく可能性。
3. 潜在的なリスクシナリオ: 警戒すべき不正のパターン（分割発注やキックバックなど）。
4. 推奨される調査アクション: 次に優先して確認すべき証憑や部門。
"""
            
            # 回答をキャッシュし、リロードのたびに課金されるのを防ぐ
            @st.cache_data(show_spinner="熟練監査人の視点で分析中...")
            def generate_ai_report(_model, _prompt):
                try:
                    response = _model.generate_content(_prompt)
                    return response.text
                except Exception as e:
                    return f"AI呼び出しエラーが発生しました。時間を置いて再試行してください: {e}"

            report = generate_ai_report(model, prompt)
            st.markdown(report)
            
            st.caption("※このレポートはAIによる仮説整理であり、不正を断定するものではありません。")

    else:
        st.error("CSVファイル内に 'amount' カラムが見つかりません。")