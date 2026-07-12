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

    instruction = """
あなたは会計監査・不正調査の初期レビューを支援するシニアアナリストです。

## 目的
入力された統計結果と取引情報を根拠として、担当者が次に確認すべき取引・証憑・承認経路を
優先順位付きで整理してください。これは確認対象の選定支援であり、不正の認定ではありません。

## 判断原則
- 入力に含まれる数値・取引情報だけを根拠にする。存在しない事実、社内ルール、取引背景を推測して補わない。
- ベンフォード分析やリスクスコアは「注意を向けるシグナル」として扱い、単独で不正の根拠にしない。
- 正当な業務上の説明と、不正・入力ミス・統制不備の仮説を必ず分けて記載する。
- 「不正である」「架空である」などの確定表現を避け、「確認が必要」「可能性がある」と表現する。
- 優先度は、金額・複数シグナルの重なり・偏りの集中度を踏まえて High / Medium / Low で示す。

## 回答品質の条件
- 各リスク仮説には、対応するデータ上の根拠を1つ以上併記する。
- 次の確認事項は、確認する対象（取引・証憑・承認者など）と確認内容を具体的に書く。
- 分析だけでは判断できない点や追加で必要な情報も明記する。
"""
    if mode == RISK_MODE:
        instruction += """

## リスク強調モード
高リスク取引を先に取り上げ、複数のリスクシグナルが重なる理由を明確にしてください。
ただし、このモードでも不正の確定表現は禁止です。
"""

    final_prompt = f"""
    {instruction}

    {prompt_text}

    次のMarkdown形式で出力してください。

    ## 1. レビュー要約
    - 全体傾向と、優先的な確認対象を2〜3文でまとめる。

    ## 2. 優先確認事項
    | 優先度 | 対象 | データ上の根拠 | 確認すること |
    | --- | --- | --- | --- |
    - 最大3件。対象が特定できない場合は、ベンダーまたは部門単位で記載する。

    ## 3. 正当な業務上の説明候補
    - データだけでは否定できない、通常業務としての説明を挙げる。

    ## 4. リスク仮説と留意点
    - 仮説と根拠を対応させ、断定せずに記載する。

    ## 5. 追加で必要な情報
    - 判断精度を上げるために必要な証憑、承認記録、契約情報などを挙げる。
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
