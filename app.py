import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import chisquare
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os

# --- 0. 日本語フォント設定（文字化け対策の核心部分） ---
# ipaexg.ttf が streamlit_app.py と同じフォルダにある前提でパスを計算
FONT_PATH = os.path.join(os.path.dirname(__file__), 'ipaexg.ttf')

# フォントファイルが存在する場合のみ、Matplotlibに適用
if os.path.exists(FONT_PATH):
    # フォントプロパティオブジェクトを作成
    jp_font_prop = fm.FontProperties(fname=FONT_PATH)
    # Matplotlibのグローバルフォントとして登録（rcParamsは古いので、この方式が確実）
    plt.rcParams['font.family'] = jp_font_prop.get_name()
    # マイナス記号の文字化け対策
    plt.rcParams['axes.unicode_minus'] = False
else:
    # ファイルがない場合はエラーを表示
    st.error(f"⚠️ フォントファイル(ipaexg.ttf)が見つかりません。GitHubリポジトリのルートにアップロードしてください。現在の検索パス: {FONT_PATH}")
    jp_font_prop = None # フォントがない場合はNoneにしておく

# --- 1. APIの設定 ---
# StreamlitのSecretsからAPIキーを取得
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop()

# モデルの読み込み（キャッシュ化して効率化）
@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        # 最新の2.5-flashを使用
        return genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )
    except Exception as e:
        st.error(f"モデルの初期化失敗: {e}")
        return None

model = get_model(GENAI_API_KEY)

# --- 2. アプリの基本UI設定 ---
st.set_page_config(page_title="会計異常検知ツール", layout="wide")
st.title("📊 会計データ：ベンフォード分布解析システム")
st.markdown("財務データの第1桁の分布を理論値と比較し、統計的な不自然さを可視化します。")
st.markdown("---")

# --- 3. サイドバー：データ管理 ---
st.sidebar.header("データソース")
uploaded_file = st.sidebar.file_uploader("解析用CSVファイルをアップロード", type="csv")

# テストデータ生成ボタン
if st.sidebar.button("🧪 テモ用データをロード"):
    np.random.seed(42)
    # 正常な対数分布データ
    normal_data = 10 ** np.random.uniform(2, 6, 2000)
    # 意図的な異常（5で始まる数値の突出）
    fraud_data = np.random.choice([500, 520, 550, 580], size=300)
    data = np.concatenate([normal_data, fraud_data])
    # セッション状態に保存
    st.session_state['data'] = pd.DataFrame({"amount": data})
    st.sidebar.success("デモデータをロードしました。")

# アップロードされたファイルをセッション状態に保存
if uploaded_file:
    st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 4. メイン解析ロジック ---
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # 必須カラムチェック
    if "amount" not in df.columns:
        st.error("CSVファイル内に 'amount' (金額)カラムが必要です。")
        st.stop()

    # データクレンジングと第1桁の抽出
    valid_amounts = pd.to_numeric(df["amount"], errors='coerce').dropna()
    # 0以下のデータはベンフォードの対象外
    valid_amounts = valid_amounts[valid_amounts > 0]
    
    # 文字列にして先頭桁を抽出
    first_digits = valid_amounts.astype(str).str.lstrip("0").str[0].astype(int)
    
    # 統計計算
    total_count = len(first_digits)
    # 各桁の出現回数をカウント
    observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    # ベンフォードの法則の理論分布（対数）
    benford_ratios = np.log10(1 + 1/np.arange(1, 10))
    expected_counts = benford_ratios * total_count
    
    # カイ二乗検定（理論分布との乖離度）
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # UI: タブ構成（日本語に戻しました）
    tab1, tab2, tab3 = st.tabs(["📊 解析結果概要", "🧠 AI分析レポート", "🔎 データエクスプローラー"])

    with tab1:
        st.subheader("分布の解析と可視化")
        col1, col2, col3 = st.columns(3)
        col1.metric("総レコード件数", f"{total_count:,} 件")
        col2.metric("p値 (有意水準 0.05)", f"{p_value:.10f}")
        
        # グラフ描画（ここが文字化け対策のポイント）
        # Streamlitのダークモードに合わせた背景色を設定
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f0f2f6')
        
        # 観測値を棒グラフで表示（透明度を上げて見やすく）
        ax.bar(range(1, 10), observed_counts / total_count, alpha=0.5, color='#1f77b4', label='観測値 (実際のデータ)')
        # 理論値を折れ線グラフで重ねる
        ax.plot(range(1, 10), benford_ratios, marker='s', color='#d62728', label='理論値 (ベンフォードの法則)', linewidth=2)
        
        # フォントが読み込めている場合のみ、日本語ラベルを適用
        if jp_font_prop:
            ax.set_xlabel("先頭桁の数字", fontproperties=jp_font_prop)
            ax.set_ylabel("出現確率", fontproperties=jp_font_prop)
            ax.legend(prop=jp_font_prop) # 凡例も日本語に
        else:
            # フォントがない場合は英語のまま（豆腐を防ぐため）
            ax.set_xlabel("Leading Digit")
            ax.set_ylabel("Frequency Ratio")
            ax.legend()
            
        st.pyplot(fig)

    with tab2:
        st.subheader("AIによる統計的違和感の分析")
        if p_value < 0.05 and model:
            # 最も乖離している桁を特定
            diffs = (observed_counts / total_count) - benford_ratios
            max_diff_digit = diffs.idxmax()
            
            # 会計士っぽさを排除した、データ科学視点のプロンプト
            prompt = f"""
あなたは調査者の思考を広げるパートナーです。
統計的な数値を『不正の証拠』として断定するのではなく、なぜその数値が偏っているのか、業務上の自然な理由（定額支払いなど）と、
注意すべきリスクの両面から仮説を提示してください。

【分析データ】
- 総件数: {total_count}
- p値: {p_value:.10f}
- 理論値から最も乖離している桁: {max_diff_digit}

【レポート構成】
1. 統計的結論: このデータ分布が自然か不自然か、p値に基づき明快に述べる。
2. 乖離の特徴: 桁「{max_diff_digit}」が突出していることの意味。
3. 想定される要因: 不正という言葉に限定せず、システム仕様（定額、端数処理など）や業務ルールの観点から推測。
4. 推奨アクション: データの信頼性を確かめるために、次にどのデータ項目（担当者、日付、相手先など）をチェックすべきか。
"""
            # キャッシュを使ってAPI呼び出しを節約
            @st.cache_data(show_spinner="AIがデータパターンを分析中...")
            def get_ai_insight(_model, _prompt):
                try:
                    return _model.generate_content(_prompt).text
                except Exception as e:
                    return f"AI呼び出しエラー: {e}"

            report_text = get_ai_insight(model, prompt)
            st.markdown(report_text)
        else:
            st.info("有意な差がないため、AI分析は不要です。")

    with tab3:
        # 有意差がある場合、乖離している桁のデータのみを表示
        observed_counts_ratio = observed_counts / total_count
        diffs = observed_counts_ratio - benford_ratios
        max_diff_digit = diffs.idxmax()
        
        st.subheader(f"重点調査対象：先頭桁が「{max_diff_digit}」のデータ")
        # amountsをastype(str)して先頭文字がmax_diff_digitのものを選ぶ
        # クレンジング後のデータのみを使うため、一度Seriesに戻す必要がある
        valid_amounts_str = valid_amounts.astype(str)
        # 0.xxxなどはlstrip("0.")してからstartswith
        is_suspicious = valid_amounts_str.str.lstrip("0.").str.startswith(str(max_diff_digit))
        
        # suspicious_dataは元のDataFrame dfをフィルタリング
        # indexがvalid_amountsと一致するものだけを対象にする
        suspicious_df = df.loc[valid_amounts[is_suspicious].index]
        
        st.write(f"該当件数: {len(suspicious_df)} 件")
        st.dataframe(suspicious_df, use_container_width=True)
        
        st.subheader("全ソースデータ")
        st.dataframe(df, use_container_width=True)

else:
    st.info("サイドバーからCSVファイルをアップロードするか、デモデータを生成してください。")