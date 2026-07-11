from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import chisquare


def generate_demo_dataset(seed=42, rows=1200):
    rng = np.random.default_rng(seed)

    departments = [
        "Sales",
        "Procurement",
        "Operations",
        "Marketing",
        "IT",
        "Finance",
        "HR",
    ]
    vendors = [
        "Northwind Supplies",
        "Apex Consulting",
        "Sakura Systems",
        "Midtown Travel",
        "BlueWave Media",
        "Vertex Tools",
        "Kitanihon Logistics",
        "Harbor Office",
        "Toyo Leasing",
        "Nimbus Services",
    ]
    employees = [
        "A. Sato",
        "K. Tanaka",
        "M. Suzuki",
        "R. Ito",
        "H. Yamada",
        "N. Kobayashi",
        "Y. Watanabe",
        "T. Kato",
        "S. Yoshida",
        "E. Nakamura",
    ]
    categories = {
        "Travel": 10000,
        "Software": 50000,
        "Office Supplies": 10000,
        "Training": 30000,
        "Equipment": 50000,
        "Meals": 10000,
    }
    payment_types = ["Corporate Card", "Bank Transfer", "Cash Reimbursement"]
    regions = ["Tokyo", "Osaka", "Nagoya", "Fukuoka"]

    dates = [
        datetime(2026, 4, 1) + timedelta(days=int(x))
        for x in rng.integers(0, 90, size=rows)
    ]
    category_choices = rng.choice(
        list(categories.keys()),
        size=rows,
        p=[0.22, 0.16, 0.2, 0.12, 0.16, 0.14],
    )
    approval_limits = np.array([categories[cat] for cat in category_choices])
    base_amounts = np.clip(np.round(10 ** rng.uniform(2.0, 4.6, size=rows), 0), 120, None)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{20260400 + i:06d}" for i in range(1, rows + 1)],
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "department": rng.choice(departments, size=rows),
            "vendor": rng.choice(vendors, size=rows),
            "employee": rng.choice(employees, size=rows),
            "category": category_choices,
            "approval_limit": approval_limits,
            "amount": base_amounts,
            "payment_type": rng.choice(payment_types, size=rows, p=[0.46, 0.44, 0.10]),
            "region": rng.choice(regions, size=rows),
        }
    )

    suspicious_idx = rng.choice(df.index, size=int(rows * 0.18), replace=False)
    suspicious_amounts = rng.choice(
        [9800, 9900, 9700, 49800, 49500, 28800, 15000],
        size=len(suspicious_idx),
        p=[0.25, 0.24, 0.14, 0.13, 0.09, 0.08, 0.07],
    )
    df.loc[suspicious_idx, "amount"] = suspicious_amounts
    df.loc[suspicious_idx, "department"] = rng.choice(
        ["Procurement", "Operations", "Marketing"],
        size=len(suspicious_idx),
        p=[0.46, 0.34, 0.20],
    )
    df.loc[suspicious_idx, "vendor"] = rng.choice(
        ["Northwind Supplies", "Vertex Tools", "Apex Consulting"],
        size=len(suspicious_idx),
        p=[0.48, 0.27, 0.25],
    )
    df.loc[suspicious_idx, "payment_type"] = rng.choice(
        ["Cash Reimbursement", "Corporate Card"],
        size=len(suspicious_idx),
        p=[0.38, 0.62],
    )

    month_end_idx = rng.choice(
        suspicious_idx, size=int(len(suspicious_idx) * 0.55), replace=False
    )
    month_end_days = rng.choice([26, 27, 28, 29, 30], size=len(month_end_idx))
    adjusted_dates = [datetime(2026, 6, int(day)).strftime("%Y-%m-%d") for day in month_end_days]
    df.loc[month_end_idx, "date"] = adjusted_dates

    exact_limit_idx = rng.choice(
        suspicious_idx, size=int(len(suspicious_idx) * 0.33), replace=False
    )
    df.loc[exact_limit_idx, "amount"] = df.loc[exact_limit_idx, "approval_limit"] - rng.choice(
        [100, 200, 300, 500],
        size=len(exact_limit_idx),
    )

    for idx in rng.choice(df.index, size=int(rows * 0.08), replace=False):
        df.loc[idx, "amount"] = rng.choice([2500, 5000, 7500, 12000, 20000, 30000])

    return df.sort_values("date").reset_index(drop=True)


def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "amount" in df.columns:
        return df, "amount"

    numeric_candidates = [
        col
        for col in df.columns
        if pd.to_numeric(df[col], errors="coerce").notna().sum() >= max(10, len(df) * 0.5)
    ]
    if numeric_candidates:
        amount_col = numeric_candidates[0]
        df = df.rename(columns={amount_col: "amount"})
        return df, amount_col

    original = df.columns[0]
    df = df.rename(columns={original: "amount"})
    return df, original


def prepare_amounts(df):
    amounts = pd.to_numeric(df["amount"], errors="coerce")
    valid = amounts.notna() & (amounts > 0)
    prepared = df.loc[valid].copy()
    prepared["amount"] = amounts.loc[valid].astype(float)
    return prepared


def add_default_columns(prepared_df):
    required_defaults = {
        "transaction_id": [f"TXN-{i:05d}" for i in range(1, len(prepared_df) + 1)],
        "date": ["2026-04-01"] * len(prepared_df),
        "department": ["Unknown"] * len(prepared_df),
        "vendor": ["Unknown"] * len(prepared_df),
        "employee": ["Unknown"] * len(prepared_df),
        "category": ["Unknown"] * len(prepared_df),
        "approval_limit": [np.nan] * len(prepared_df),
        "payment_type": ["Unknown"] * len(prepared_df),
    }
    for col, default in required_defaults.items():
        if col not in prepared_df.columns:
            prepared_df[col] = default
    return prepared_df


def first_digits_from_amounts(amounts):
    normalized = amounts.astype(str).str.lstrip("0.")
    return normalized.str[0].astype(int)


def benford_analysis(df):
    first_digits = first_digits_from_amounts(df["amount"])
    total = len(first_digits)
    observed = first_digits.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    expected_ratio = np.log10(1 + 1 / np.arange(1, 10))
    chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected_ratio * total)
    actual_ratio = observed / total
    deviation = actual_ratio - expected_ratio
    anomaly_digit = int(deviation.idxmax())
    mad = float(np.mean(np.abs(actual_ratio.values - expected_ratio)))

    return {
        "total": total,
        "observed": observed,
        "expected_ratio": expected_ratio,
        "actual_ratio": actual_ratio,
        "p_value": float(p_value),
        "chi_stat": float(chi_stat),
        "mad": mad,
        "deviation": deviation,
        "anomaly_digit": anomaly_digit,
    }


def categorize_anomaly(p_value, mad):
    if p_value < 0.01 or mad > 0.02:
        return "高", "#dc2626"
    if p_value < 0.05 or mad > 0.012:
        return "中", "#d97706"
    return "低", "#0f766e"


def enrich_risk_signals(df, anomaly_digit, deviation):
    detailed = df.copy()
    detailed["date"] = pd.to_datetime(detailed["date"], errors="coerce")
    detailed["first_digit"] = first_digits_from_amounts(detailed["amount"])
    detailed["digit_gap_pct"] = detailed["first_digit"].map(
        {digit: float(max(gap, 0)) * 100 for digit, gap in deviation.items()}
    )
    detailed["near_approval_limit"] = (
        detailed["approval_limit"].notna()
        & (detailed["amount"] >= detailed["approval_limit"] * 0.9)
        & (detailed["amount"] < detailed["approval_limit"])
    )
    detailed["round_amount"] = np.isclose(detailed["amount"] % 100, 0, atol=1)
    detailed["repeat_count"] = detailed.groupby("amount")["amount"].transform("size")
    detailed["repeat_amount"] = detailed["repeat_count"] >= 4
    detailed["month_end"] = detailed["date"].dt.day.fillna(0).astype(int) >= 26
    detailed["cash_related"] = detailed["payment_type"].fillna("").eq("Cash Reimbursement")
    detailed["anomalous_digit"] = detailed["first_digit"].eq(anomaly_digit)

    detailed["risk_score"] = (
        detailed["anomalous_digit"].astype(int) * 28
        + detailed["near_approval_limit"].astype(int) * 24
        + detailed["round_amount"].astype(int) * 12
        + detailed["repeat_amount"].astype(int) * 14
        + detailed["month_end"].astype(int) * 10
        + detailed["cash_related"].astype(int) * 7
        + detailed["digit_gap_pct"].clip(0, 15)
    )
    detailed["risk_score"] = detailed["risk_score"].clip(0, 100).round(0).astype(int)

    reason_columns = {
        "Benford偏差の大きい桁に一致": detailed["anomalous_digit"],
        "承認上限の直前": detailed["near_approval_limit"],
        "キリの良い金額": detailed["round_amount"],
        "同額取引が多発": detailed["repeat_amount"],
        "月末集中": detailed["month_end"],
        "現金精算": detailed["cash_related"],
    }
    reason_texts = []
    for idx in detailed.index:
        reasons = [label for label, mask in reason_columns.items() if bool(mask.loc[idx])]
        reason_texts.append(" / ".join(reasons[:3]) if reasons else "追加確認ポイントなし")
    detailed["risk_reasons"] = reason_texts

    bins = [-1, 34, 59, 100]
    labels = ["低", "中", "高"]
    detailed["risk_band"] = pd.cut(detailed["risk_score"], bins=bins, labels=labels)

    return detailed.sort_values(["risk_score", "amount"], ascending=[False, False]).reset_index(drop=True)


def build_hotspots(detailed_df):
    vendor_view = (
        detailed_df.groupby("vendor")
        .agg(
            transactions=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            avg_risk=("risk_score", "mean"),
            high_risk=("risk_band", lambda s: int((s == "高").sum())),
        )
        .sort_values(["high_risk", "avg_risk"], ascending=False)
        .reset_index()
    )

    department_view = (
        detailed_df.groupby("department")
        .agg(
            transactions=("transaction_id", "count"),
            total_amount=("amount", "sum"),
            avg_risk=("risk_score", "mean"),
            high_risk=("risk_band", lambda s: int((s == "高").sum())),
        )
        .sort_values(["high_risk", "avg_risk"], ascending=False)
        .reset_index()
    )

    reason_view = (
        detailed_df.assign(primary_reason=detailed_df["risk_reasons"].str.split(" / ").str[0])
        .groupby("primary_reason")
        .agg(transactions=("transaction_id", "count"), avg_risk=("risk_score", "mean"))
        .sort_values("transactions", ascending=False)
        .reset_index()
    )

    return vendor_view, department_view, reason_view


def high_risk_signature(detailed_df):
    top = detailed_df.head(10)[["transaction_id", "risk_score"]].astype(str)
    return "|".join((top["transaction_id"] + ":" + top["risk_score"]).tolist())
