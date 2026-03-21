import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. API & モデル設定 (404対策) ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' を設定してください。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        # 2026年現在の安定版フルネームを指定（404回避）
        return genai.GenerativeModel(
            model_name='models/gemini-1.5-flash-002', 
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception:
        return genai.GenerativeModel(model_name='models/gemini-pro')

model = get_model(GENAI_API_KEY)

# キャッシュの引数から _ を削除し、モード切替を即座に反映させる
@st.cache_data(show_spinner="AI分析中...")
def get_ai_insight(_model, prompt_text):
    try:
        return _model.generate_content(prompt_text).text
    except Exception as e:
        return f"AI呼び出しエラー: {e}"

# --- 2. UI構成 ---
st.set_page_config(page_title="会計異常検知", layout="wide")
st.title("📊 ベンフォード分布解析システム")

st.sidebar.header("設定")
uploaded_file = st.sidebar.file_uploader("CSVアップロード", type="csv")
if st.sidebar.button("🧪 デモデータ生成"):
    np.random.seed(42)
    # 意図的に「9」を増やしたデータ
    data = np.concatenate([10**np.random.uniform(2, 5, 1000), np.random.choice([900, 950, 990], 200)])
    st.session_state['data'] = pd.DataFrame({"amount": data})

tone_mode = st.sidebar.radio("AI回答スタイル", ["理解支援モード", "断定モード"])

# --- 3. 解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    amounts = amounts[amounts > 0]
    # 先頭桁の抽出
    first_digits = amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    
    total = len(first_digits)
    obs = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    exp_ratios = np.log10(1 + 1/np.arange(1, 10))
    _, p_val = chisquare(f_obs=obs, f_exp=exp_ratios * total)

    t1, t2 = st.tabs(["📊 統計グラフ", "🧠 AIレポート"])

    with t1:
        st.metric("p-value", f"{p_val:.10f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, 10), obs/total, alpha=0.4, label="Observed", color="skyblue")
        ax.plot(range(1, 10), exp_ratios, 'ro-', label="Benford Line")
        ax.set_xlabel("Leading Digit")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    with t2:
        if p_val < 0.05:
            max_digit = ((obs/total) - exp_ratios).idxmax()
            
            # 断定しないプロンプトの肝
            if tone_mode == "理解支援モード":
                instr = "あなたは『思考の伴走者』です。統計的乖離を不正と断定せず、『〜の可能性』『〜という視点』と記述し、業務上の正当な理由（価格設定など）の仮説を必ず含めてください。"
            else:
                instr = "あなたは厳しい監査人です。断定的に異常を指摘し、不正の兆候として警告してください。"

            prompt = f"{instr}\n分析データ: 総数{total}, p値{p_val:.6f}, 最も乖離した桁{max_digit}\n構成: 1.観測事実 2.背景考察 3.業務要因の仮説 4.推奨アクション"
            st.markdown(get_ai_insight(model, prompt))
        else:
            st.success("統計的に有意な偏りは認められませんでした。")
else:
    st.info("サイドバーからデータをロードしてください。")