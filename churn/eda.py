"""Exploratory data analysis: saves charts to reports/figures and stats to reports/eda_summary.md.

Usage: python -m churn.eda [--use-sample]
"""
import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config
from .data import load_clean, load_raw, resolve_data_path

sns.set_theme(style="whitegrid")
T = config.TARGET


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def churn_rate_bar(df, col, name, title, order=None):
    rate = df.groupby(col)[T].mean().mul(100)
    rate = rate.reindex(order) if order else rate.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#3b78c4")
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.set(title=title, xlabel=col, ylabel="Churn rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, name)


def main(use_sample=False):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    path = resolve_data_path(use_sample)
    raw = load_raw(path)
    df, _ = load_clean(use_sample)

    blank_total = int(pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum())
    summary = [
        "# EDA summary (computed from the dataset)\n",
        f"- Source file: `{path.relative_to(config.ROOT)}`" + ("  **(SYNTHETIC)**" if use_sample else ""),
        f"- Rows: {len(raw)} | Columns: {raw.shape[1]} | Duplicate customerIDs: {int(raw[config.ID_COLUMN].duplicated().sum())}",
        f"- Blank/invalid TotalCharges: {blank_total} (typically customers with tenure 0)",
        f"- Churn rate: {df[T].mean():.1%} ({int(df[T].sum())} churned / {len(df)})",
        f"- Missing values per column after cleaning: {df.isna().sum()[df.isna().sum() > 0].to_dict() or 'none'}\n",
        "## Churn rate by segment (%)\n",
    ]
    for col in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "TechSupport", "PaperlessBilling"]:
        summary.append(f"**{col}**\n\n" + df.groupby(col)[T].mean().mul(100).round(1).to_frame("churn_%").to_markdown() + "\n")
    df["TenureBand"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    summary.append("**Tenure band (months)**\n\n" + df.groupby("TenureBand", observed=True)[T].mean().mul(100).round(1)
                   .to_frame("churn_%").to_markdown() + "\n")
    summary.append("## Numeric summary\n\n" + df[config.NUMERIC_FEATURES].describe().round(2).to_markdown() + "\n")
    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(summary))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[T].map({0: "Stayed", 1: "Churned"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#3b78c4", "#d9534f"], hue=counts.index, legend=False)
    ax.set(title="Target distribution (class balance)", xlabel="", ylabel="Customers")
    _save(fig, "01_target_distribution.png")

    churn_rate_bar(df, "Contract", "02_churn_by_contract.png", "Churn rate by contract type",
                   ["Month-to-month", "One year", "Two year"])
    churn_rate_bar(df, "InternetService", "03_churn_by_internet_service.png", "Churn rate by internet service")
    churn_rate_bar(df, "PaymentMethod", "04_churn_by_payment_method.png", "Churn rate by payment method")
    churn_rate_bar(df, "TenureBand", "05_churn_by_tenure_band.png", "Churn rate by tenure band (months)",
                   ["0-12", "13-24", "25-48", "49-72"])

    labelled = df.assign(Status=df[T].map({0: "Stayed", 1: "Churned"}))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, unit in zip(axes, config.NUMERIC_FEATURES, ["months", "$ / month", "$ total"]):
        sns.kdeplot(data=labelled, x=col, hue="Status", fill=True, common_norm=False, ax=ax)
        ax.set(title=f"{col} by churn status", xlabel=f"{col} ({unit})")
    _save(fig, "06_numeric_distributions.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    corr = df[config.NUMERIC_FEATURES + [T]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation (numeric features + churn)")
    _save(fig, "07_correlation_heatmap.png")
    logging.info("EDA complete: %s", config.FIGURES_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-sample", action="store_true")
    main(ap.parse_args().use_sample)
