import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. API & モデル設定 (404を徹底的に調査するロジック) ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' を設定してください。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        # 利用可能な全モデルを取得
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 画面に利用可能なモデルを表示（デバッグ用：動いたら消してOK）
        if not available_models:
            st.error("利用可能なモデルが一つも見つかりません。APIキーの権限を確認してください。")
            return None
            
        # 優先順位をつけて選択
        target = None
        for candidate in ["models/gemini-1.5-flash", "models/gemini-1.5-flash-001", "models/gemini-pro"]:
            if candidate in available_models:
                target = candidate
                break
        
        # 候補がなければ、リストの一番最初を強制的に使う
        if not target:
            target = available_models[0]
            
        st.sidebar.info(f"使用中モデル: {target}")
        
        return genai.GenerativeModel(
            model_name=target,
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        st.error(f"モデル初期化の致命的エラー: {e}")
        return None

model = get_model(GENAI_API_KEY)

@st.cache_data(show_spinner="AI分析中...")
def get_ai_insight(_model, prompt_text):
    if _model is None: return "モデルが初期化されていないため分析できません。"
    try:
        return _model.generate_content(prompt_text).text
    except Exception as e:
        return f"AI呼び出しエラー (内容を確認してください): {e}"

# --- 2. UI & 解析 (中略なしの全コード) ---
st.set_page_config(page_title="会計異常検知", layout="wide")
st.title("📊 ベンフォード分布解析システム")

st.sidebar.header("設定")
uploaded_file = st.sidebar.file_uploader("CSVアップロード", type="csv")
if st.sidebar.button("🧪 デモデータ生成"):
    np.random.seed(42)
    data = np.concatenate([10**np.random.uniform(2, 5, 1000), np.random.choice([900, 950], 200)])
    st.session_state['data'] = pd.DataFrame({"amount": data})

tone_mode = st.sidebar.radio("AI回答スタイル", ["理解支援モード", "断定モード"])

if 'data' in st.session_state:
    df = st.session_state['data']
    amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    amounts = amounts[amounts > 0]
    first_digits = amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    
    total = len(first_digits)
    obs = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    exp_ratios = np.log10(1 + 1/np.arange(1, 10))
    _, p_val = chisquare(f_obs=obs, f_exp=exp_ratios * total)

    t1, t2 = st.tabs(["📊 統計グラフ", "🧠 AIレポート"])

    with t1:
        st.metric("p-value", f"{p_val:.10f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, 10), obs/total, alpha=0.4, label="Observed")
        ax.plot(range(1, 10), exp_ratios, 'ro-', label="Benford Line")
        st.pyplot(fig)

    with t2:
        if p_val < 0.05:
            max_digit = ((obs/total) - exp_ratios).idxmax()
            if tone_mode == "理解支援モード":
                instr = "断定を避け、業務上の正当な理由の仮説を必ず含めて『思考の伴走者』として回答してください。"
            else:
                instr = "厳しい監査人として、断定的に異常を指摘してください。"

            prompt = f"{instr}\nデータ: 総数{total}, p値{p_val:.6f}, 乖離桁{max_digit}"
            st.markdown(get_ai_insight(model, prompt))
        else:
            st.success("有意な偏りは認められませんでした。")
else:
    st.info("データをロードしてください。")