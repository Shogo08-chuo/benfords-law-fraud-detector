import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

# --- 1. API & モデル設定 (2.5-flash対応・404完全回避) ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' を設定してください。")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 最新の2.5-flashか1.5-flashを自動選択
        target = next((m for m in models if "2.5-flash" in m), None) or \
                 next((m for m in models if "1.5-flash" in m), models[0])
        return genai.GenerativeModel(
            model_name=target, 
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        st.error(f"モデル接続エラー: {e}")
        return None

model = get_model(GENAI_API_KEY)

@st.cache_data(show_spinner="LLMが解釈支援レポートを生成中...")
def get_ai_insight(_model, prompt_text):
    try:
        return _model.generate_content(prompt_text).text
    except Exception as e:
        return f"AI呼び出しエラー: {e}"

# --- 2. セッション状態の初期化（実験H3用タイマーなど） ---
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = None
if 'elapsed_time' not in st.session_state:
    st.session_state['elapsed_time'] = None

# --- 3. UI構成とサイドバー ---
st.set_page_config(page_title="会計異常検知フレームワーク", layout="wide")
st.title("📊 会計不正の理解支援フレームワーク")
st.markdown("本システムは、**「統計による違和感検知」と「LLMによる意味理解支援」を分離**し、調査者の判断をサポートする研究用プロトタイプです。")

st.sidebar.header("📁 1. データ設定")
uploaded_file = st.sidebar.file_uploader("CSVアップロード", type="csv")
if st.sidebar.button("🧪 デモデータ生成 (異常値含む)"):
    np.random.seed(42)
    data = np.concatenate([10**np.random.uniform(2, 5, 2000), np.random.choice([900, 950, 990], 300)])
    st.session_state['data'] = pd.DataFrame({"amount": data})

st.sidebar.header("⚙️ 2. 実験条件 (H4検証用)")
tone_mode = st.sidebar.radio("LLMの提示スタイル", ["理解支援モード (提案型)", "断定モード (警告型)"])

st.sidebar.header("⏱️ 3. 調査計測 (H3検証用)")
if st.sidebar.button("▶️ 調査開始 (タイマーON)"):
    st.session_state['start_time'] = time.time()
    st.session_state['elapsed_time'] = None
if st.sidebar.button("⏹️ 調査終了 (タイマーOFF)"):
    if st.session_state['start_time']:
        st.session_state['elapsed_time'] = time.time() - st.session_state['start_time']
        st.session_state['start_time'] = None

if st.session_state['start_time']:
    st.sidebar.warning("⏳ 調査実行中...")
elif st.session_state['elapsed_time']:
    st.sidebar.success(f"✅ 調査時間: {st.session_state['elapsed_time']:.1f} 秒")

# --- 4. メイン解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    amounts = amounts[amounts > 0]
    first_digits = amounts.astype(str).str.lstrip("0.").str[0].astype(int)
    
    total = len(first_digits)
    obs = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    exp_ratios = np.log10(1 + 1/np.arange(1, 10))
    _, p_val = chisquare(f_obs=obs, f_exp=exp_ratios * total)

    # 論文の設計思想に基づくタブ構成
    tab1, tab2, tab3, tab4 = st.tabs([
        "第一段階：違和感検知", 
        "第二段階：意味理解支援", 
        "実務的絞り込み", 
        "📝 評価実験フォーム"
    ])

    with tab1:
        st.subheader("第一段階：数値データに内在する統計的違和感の検知")
        st.markdown("この段階では「不正」の判定は行わず、純粋な統計的乖離のみを可視化します。（再現性の担保）")
        col1, col2 = st.columns(2)
        col1.metric("分析対象レコード数", f"{total:,} 件")
        col2.metric("ベンフォード分布との乖離 (p値)", f"{p_val:.10f}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, 10), obs/total, alpha=0.5, label="観測値", color="#1f77b4")
        ax.plot(range(1, 10), exp_ratios, 'ro-', label="理論値 (ベンフォード則)", linewidth=2)
        ax.set_xlabel("先頭桁 (Leading Digit)")
        ax.set_ylabel("出現確率")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("第二段階：LLMを用いた違和感の意味理解支援")
        if p_val < 0.05 and model:
            max_digit = ((obs/total) - exp_ratios).idxmax()
            
            # --- 研究の核心：理解支援 vs 断定のプロンプト制御 ---
            if tone_mode == "理解支援モード (提案型)":
                instr = """あなたは調査者の判断を支援する思考の伴走者です。
                【厳守事項】
                1. 統計的乖離を「不正」と断定せず、「〜の可能性」という表現に留める。
                2. 仮説の多様性を広げるため、「想定される正当な業務上の理由」と「想定される不正シナリオ」の両方を必ず並列で提示する。
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
            - 統計的有意差(p値): {p_val:.6f}
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
            st.markdown(f"統計分析とLLMの考察に基づき、最も違和感の強い **先頭桁「{max_digit}」** のデータを抽出しました。")
            
            target_df = df[df["amount"].astype(str).str.lstrip("0.").str.startswith(str(max_digit))]
            st.metric(f"桁「{max_digit}」の該当件数", f"{len(target_df)} 件")
            st.dataframe(target_df, use_container_width=True)
            
            st.download_button(
                label="この調査対象リストをCSVでダウンロード",
                data=target_df.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"suspicious_data_digit_{max_digit}.csv"
            )

    with tab4:
        st.subheader("📝 研究用：仮説検証アンケート (H1, H2, H4)")
        st.markdown("調査終了後、以下のフォームに回答してください。")
        with st.form("evaluation_form"):
            q1 = st.slider("【H1】LLMの説明により、調査すべき箇所が明確になりましたか？ (1:全く思わない - 5:強く思う)", 1, 5, 3)
            q2 = st.slider("【H2】LLMの説明により、自分が思いつかなかった新しい仮説（正当理由・不正シナリオ）に気づけましたか？", 1, 5, 3)
            q4 = st.slider("【H4】LLMの出力内容は、調査の裏付けとして信頼できると感じましたか？", 1, 5, 3)
            submitted = st.form_submit_button("評価データを記録")
            if submitted:
                st.success(f"記録完了！ (H1: {q1}, H2: {q2}, H4: {q4}, 所要時間: {st.session_state['elapsed_time']}秒)")
else:
    st.info("👈 サイドバーからCSVをアップロードするか、デモデータを生成してください。")