import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import requests
from datetime import datetime
import plotly.graph_objects as go

# ==========================================
# 1. 初期設定とAIモデルの準備
# ==========================================
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("⚠️ Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = next((m for m in models if "2.5-flash" in m), None) or \
                 next((m for m in models if "1.5-flash" in m), models[0])
        return genai.GenerativeModel(
            model_name=target, 
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        st.error(f"モデルの接続に失敗しました: {e}")
        return None

model = get_model(GENAI_API_KEY)

@st.cache_data(show_spinner="LLMが解釈支援レポートを生成中...")
def get_ai_insight(_model, prompt_text):
    try:
        return _model.generate_content(prompt_text).text
    except Exception as e:
        return f"AI呼び出しエラーが発生しました: {e}"

# ==========================================
# 2. セッション状態（タイマー機能）の初期化
# ==========================================
if 'start_time' not in st.session_state: 
    st.session_state['start_time'] = None
if 'elapsed_time' not in st.session_state: 
    st.session_state['elapsed_time'] = None

# ==========================================
# 3. UIレイアウトとサイドバー
# ==========================================
st.set_page_config(page_title="会計不正の理解支援フレームワーク", layout="wide")
st.title("📊 会計データ：意味理解支援フレームワーク")

st.sidebar.header("📁 1. データ設定")
uploaded_file = st.sidebar.file_uploader("CSVアップロード", type="csv")
if st.sidebar.button("🧪 デモデータ生成"):
    np.random.seed(42)
    data = np.concatenate([10**np.random.uniform(2, 5, 2000), np.random.choice([900, 950, 990], 300)])
    st.session_state['data'] = pd.DataFrame({"amount": data})

st.sidebar.header("⚙️ 2. 実験条件 (H4検証用)")
tone_mode = st.sidebar.radio("LLMの提示スタイル", ["理解支援モード (提案型)", "断定モード (警告型)"])

st.sidebar.header("⏱️ 3. 調査計測 (H3検証用)")
if st.sidebar.button("▶️ 調査開始"):
    st.session_state['start_time'] = time.time()
    st.session_state['elapsed_time'] = None

if st.sidebar.button("⏹️ 調査終了"):
    if st.session_state['start_time']:
        st.session_state['elapsed_time'] = time.time() - st.session_state['start_time']
        st.session_state['start_time'] = None

if st.session_state['start_time']:
    st.sidebar.warning("⏳ 調査実行中...")
elif st.session_state['elapsed_time']:
    st.sidebar.success(f"✅ 調査完了: {st.session_state['elapsed_time']:.1f} 秒")

# ==========================================
# 4. メイン解析とタブ表示
# ==========================================
if 'data' in st.session_state:
    df = st.session_state['data']
    amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    amounts = amounts[amounts > 0]
    first_digits = amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    
    total = len(first_digits)
    obs = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    exp_ratios = np.log10(1 + 1/np.arange(1, 10))
    _, p_val = chisquare(f_obs=obs, f_exp=exp_ratios * total)

    tab1, tab2, tab3, tab4 = st.tabs([
        "第一段階：検知", 
        "第二段階：理解支援", 
        "実務的絞り込み", 
        "📝 評価実験フォーム"
    ])

    with tab1:
        st.subheader("第一段階：統計的違和感の検知")
        st.markdown("不正の判定は行わず、純粋な統計的乖離のみを可視化します。")
        st.metric("p-value (有意水準0.05)", f"{p_val:.10f}")
        
        # --- Plotlyによる文字化けしない美しいグラフ ---
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, 10)), y=obs/total,
            name="観測値 (実際のデータ)", marker_color="#1f77b4", opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, 10)), y=exp_ratios,
            mode="lines+markers", name="理論値 (ベンフォードの法則)",
            line=dict(color="red", width=2), marker=dict(size=8)
        ))
        fig.update_layout(
            xaxis_title="先頭桁 (Leading Digit)", yaxis_title="出現確率",
            xaxis=dict(tickmode='linear'), hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("第二段階：LLMによる意味理解支援")
        if p_val < 0.05 and model:
            max_digit = ((obs/total) - exp_ratios).idxmax()
            
            if tone_mode == "理解支援モード (提案型)":
                instr = """あなたは調査者の判断を支援する「思考の伴走者」です。
                【厳守事項】
                1. 統計的乖離を「不正」と断定せず、「〜の可能性」という表現に留める。
                2. 仮説の多様性を広げるため、「想定される正当な業務上の理由」と「想定される不正シナリオ」の両方を必ず提示する。
                3. 判断は調査者に委ねる姿勢を貫く。"""
            else:
                instr = """あなたは厳しい監査人です。
                【厳守事項】
                1. 統計的乖離を不正の明らかな兆候として断定的に警告する。
                2. 他の可能性（正当な理由など）は考慮せず、リスクのみを強調する。"""

            prompt = f"""
            {instr}
            
            【検知された違和感データ】
            - 分析件数: {total}件
            - p値: {p_val:.6f}
            - 最も異常に突出している桁: 「{max_digit}」
            
            【出力構成】
            1. 観測事実の要約
            2. 想定される正当な理由（システム制約、価格設定ルールなど）
            3. 想定される不正・エラーのシナリオ
            4. 調査者が次に確認すべき情報（推奨アクション）
            """
            st.markdown(get_ai_insight(model, prompt))
        else:
            st.success("統計的に有意な偏りは認められませんでした。")

    with tab3:
        st.subheader("実務的な調査対象の絞り込み")
        if p_val < 0.05:
            max_digit = ((obs/total) - exp_ratios).idxmax()
            st.markdown(f"最も違和感の強い **先頭桁「{max_digit}」** のデータを抽出しました。")
            
            target_df = df[df["amount"].astype(str).str.lstrip("0.").str.startswith(str(max_digit))]
            st.metric(f"桁「{max_digit}」の該当件数", f"{len(target_df)} 件")
            st.dataframe(target_df, use_container_width=True)

    with tab4:
        st.subheader("📝 研究用：仮説検証アンケート (自動保存)")
        st.markdown("調査終了後、以下のフォームに回答してください。データはスプレッドシートに直接保存されます。")
        
        with st.form("evaluation_form"):
            q1 = st.slider("【H1】LLMの説明により、調査すべき箇所が明確になりましたか？ (1:全く思わない - 5:強く思う)", 1, 5, 3)
            q2 = st.slider("【H2】LLMの説明により、自分が思いつかなかった新しい仮説に気づけましたか？", 1, 5, 3)
            q4 = st.slider("【H4】LLMの出力内容は、調査の裏付けとして信頼できると感じましたか？", 1, 5, 3)
            submitted = st.form_submit_button("評価データを記録")
            
            if submitted:
                GAS_URL = "https://script.google.com/macros/s/AKfycbzm0u7NwHeMlHNJ7dP2XgMBx8ZnQVyTI1nHuhB2Zoiibcf63tjCD7ojxbuC-v-ZIij7WQ/exec"
                
                elapsed = st.session_state.get('elapsed_time')
                final_time = round(elapsed, 1) if elapsed else 0
                
                payload = {
                    "date": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "style": str(tone_mode),
                    "q1": str(q1),
                    "q2": str(q2),
                    "time": str(final_time),
                    "q4": str(q4)
                }
                
                try:
                    response = requests.post(GAS_URL, data=payload)
                    if response.status_code == 200 and "Success" in response.text:
                        st.success(f"🎉 記録大成功！スプレッドシートにデータが書き込まれました。(所要時間: {final_time}秒)")
                        st.balloons()
                    else:
                        st.error(f"⚠️ 送信に失敗しました。ステータス: {response.status_code}")
                except Exception as e:
                    st.error(f"⚠️ 通信エラーが発生しました: {e}")
else:
    st.info("👈 サイドバーからCSVをアップロードするか、デモデータを生成してください。")