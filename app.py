import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, skew, kurtosis
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. API Configuration ---
if "GEMINI_API_KEY" in st.secrets:
    GENAI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Error: GEMINI_API_KEY not found in secrets.")
    st.stop()

@st.cache_resource
def get_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name='gemini-2.0-flash', # 最新の安定版を指定
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        return None

model = get_model(GENAI_API_KEY)

# --- 2. App Layout Configuration ---
st.set_page_config(page_title="Forensic Accounting Analyzer", layout="wide")
st.title("Digital Forensic: Benford's Law Analysis System")

# --- 3. Sidebar: Data Controls ---
with st.sidebar:
    st.header("Data Source")
    if st.button("Generate Synthetic Fraud Data"):
        np.random.seed(42)
        normal = 10 ** np.random.uniform(2, 6, 2500)
        # 特定の数値を意図的に注入（不正シミュレーション）
        fraud = np.random.choice([512, 528, 550, 589], size=400)
        data = np.concatenate([normal, fraud])
        st.session_state['data'] = pd.DataFrame({"amount": data})
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")
    if uploaded_file:
        st.session_state['data'] = pd.read_csv(uploaded_file)

# --- 4. Analysis Logic ---
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # 金額列の自動判定（"amount" または最初の数値列）
    target_col = "amount" if "amount" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    
    # 前処理: 数値抽出と先頭桁の取得
    clean_series = pd.to_numeric(df[target_col], errors='coerce').dropna()
    clean_series = clean_series[clean_series > 0]
    
    first_digits = clean_series.astype(str).str.lstrip("0").str[0].astype(int)
    
    # 統計計算
    observed_counts = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    total_samples = len(first_digits)
    observed_freq = observed_counts / total_samples
    
    benford_dist = np.log10(1 + 1/np.arange(1, 10))
    expected_counts = benford_dist * total_samples
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # --- 5. Metrics & Visualization ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Statistical Summary")
        st.metric("Total Samples", f"{total_samples:,}")
        st.metric("Chi-Square Statistic", f"{chi_stat:.4f}")
        st.metric("P-Value", f"{p_value:.8f}")
        
        # 追加統計量
        st.markdown("---")
        st.write("**Descriptive Statistics**")
        stats_df = pd.DataFrame({
            "Metric": ["Mean", "Median", "Skewness", "Kurtosis"],
            "Value": [clean_series.mean(), clean_series.median(), skew(clean_series), kurtosis(clean_series)]
        })
        st.table(stats_df)

    with col2:
        st.subheader("Frequency Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(1, 10)
        ax.bar(x, observed_freq, color='#1f77b4', alpha=0.7, label='Observed (Sample)')
        ax.plot(x, benford_dist, marker='o', linestyle='--', color='#d62728', label='Theoretical (Benford)')
        ax.set_xticks(x)
        ax.set_xlabel("First Digit")
        ax.set_ylabel("Probability")
        ax.legend()
        st.pyplot(fig)

    # --- 6. Deviation Analysis & Deep Dive ---
    st.divider()
    st.subheader("Anomalous Value Detection")
    
    # 乖離率の計算
    deviation = (observed_freq - benford_dist) / benford_dist
    max_dev_digit = deviation.idxmax()
    
    dev_col1, dev_col2 = st.columns(2)
    
    with dev_col1:
        st.write(f"Digit with Maximum Deviation: **{max_dev_digit}**")
        # 異常が疑われる行を抽出
        anomaly_rows = df[first_digits == max_dev_digit]
        st.write(f"Transactions starting with '{max_dev_digit}': {len(anomaly_rows)} records")
        st.dataframe(anomaly_rows.head(10), use_container_width=True)

    with dev_col2:
        if model:
            st.write("**Expert Analysis Hypothesis**")
            prompt = f"""
            Analyze the following forensic audit results:
            - Dataset Size: {total_samples}
            - P-value: {p_value:.8f}
            - Most Deviated Digit: {max_dev_digit} (Deviation: {deviation[max_dev_digit]:.2%})
            - Skewness: {skew(clean_series):.2f}
            
            Provide a concise audit report including:
            1. Risk assessment based on p-value.
            2. Potential reasons for the deviation in digit {max_dev_digit}.
            3. Recommended follow-up procedures.
            Language: Japanese. Avoid emojis.
            """
            
            if st.button("Generate Audit Report"):
                with st.spinner("Analyzing..."):
                    response = model.generate_content(prompt)
                    st.info(response.text)

else:
    st.info("CSVファイルをアップロードするか、デモ用データを生成してください。")