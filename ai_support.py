import os

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import HarmBlockThreshold, HarmCategory


BALANCED_MODE = "理解支援モード"
RISK_MODE = "リスク強調モード"


@st.cache_resource
def get_model():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "Gemini APIキーが未設定のため、AI解釈は無効です。"

    try:
        genai.configure(api_key=api_key)
        models = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        target = next((m for m in models if "2.5-flash" in m), None) or next(
            (m for m in models if "1.5-flash" in m),
            models[0],
        )
        model = genai.GenerativeModel(
            model_name=target,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            },
        )
        return model, None
    except Exception as exc:
        return None, f"Gemini接続エラー: {exc}"


@st.cache_data(show_spinner="AIが確認ポイントを整理中...")
def get_ai_insight(prompt_text, mode):
    model, error = get_model()
    if error or model is None:
        return None, error or "AIモデルを利用できません。"

    instruction = (
        "あなたは監査担当者の判断を支援するアナリストです。"
        "統計的違和感を不正と断定せず、正当な業務理由と不正・エラー仮説を分けて提示し、"
        "確認すべき観点を優先順位付きで提案してください。"
    )
    if mode == RISK_MODE:
        instruction = (
            "あなたは厳格なリスクレビュー担当です。"
            "統計的違和感を強いリスクシグナルとして扱い、リスクの高い仮説と確認ポイントを優先して提示してください。"
            "ただし不正の確定表現は避けてください。"
        )

    final_prompt = f"""
    {instruction}

    {prompt_text}

    出力形式:
    1. 要約
    2. 正当な理由の候補
    3. リスクの候補
    4. 次に確認すること
    """

    try:
        return model.generate_content(final_prompt).text, None
    except Exception as exc:
        return None, f"AI呼び出しエラー: {exc}"


def create_ai_prompt(analysis, detailed_df, vendor_view, department_view):
    top_cases = detailed_df.head(5)[
        ["transaction_id", "vendor", "department", "amount", "risk_score", "risk_reasons"]
    ]
    vendor_lines = [
        f"- {row.vendor}: 高リスク {row.high_risk}件, 平均スコア {row.avg_risk:.1f}"
        for row in vendor_view.head(3).itertuples()
    ]
    dept_lines = [
        f"- {row.department}: 高リスク {row.high_risk}件, 平均スコア {row.avg_risk:.1f}"
        for row in department_view.head(3).itertuples()
    ]
    case_lines = [
        f"- {row.transaction_id} / {row.vendor} / {row.department} / {row.amount:,.0f} / score {row.risk_score} / {row.risk_reasons}"
        for row in top_cases.itertuples()
    ]

    return f"""
    【統計的違和感】
    - 分析対象件数: {analysis["total"]}件
    - p値: {analysis["p_value"]:.6f}
    - χ²統計量: {analysis["chi_stat"]:.3f}
    - MAD: {analysis["mad"]:.4f}
    - 最も理論値から乖離している先頭桁: {analysis["anomaly_digit"]}

    【ベンダー別の偏り】
    {chr(10).join(vendor_lines)}

    【部門別の偏り】
    {chr(10).join(dept_lines)}

    【優先して確認したい取引】
    {chr(10).join(case_lines)}
    """


def get_gas_url():
    return st.secrets.get("GAS_URL") or os.getenv("GAS_URL")
