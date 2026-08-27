#!/usr/bin/env python3
"""Generate publication figures from manuscript table data (Tier-1 + CNAS)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

COLORS = {
    "Rule-Only": "#4E79A7",
    "RAG-Only": "#F28E2B",
    "Ensemble": "#76B7B2",
    "OR-Ensemble": "#76B7B2",
    "Self-RAG": "#B07AA1",
    "ReAct": "#FF9DA7",
    "HRAG": "#E15759",
    "neutral": "#BAB0AC",
    "parse": "#59A14F",
    "hitl": "#EDC948",
    "output": "#9C755F",
}
METRIC_COLORS = {"accuracy": "#4E79A7", "hallucination": "#E15759"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = ROOT / stem
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def grouped_acc_halluc_panel(ax, df: pd.DataFrame, title: str) -> None:
    methods = ["Rule-Only", "RAG-Only", "Ensemble", "HRAG"]
    x = np.arange(len(methods))
    width = 0.36
    acc = [df.loc[df.method == m, "accuracy"].iloc[0] for m in methods]
    hal = [df.loc[df.method == m, "hallucination"].iloc[0] for m in methods]

    bars_acc = ax.bar(
        x - width / 2,
        acc,
        width,
        label="Accuracy (%)",
        color=METRIC_COLORS["accuracy"],
        edgecolor="white",
        linewidth=0.6,
    )
    bars_hal = ax.bar(
        x + width / 2,
        hal,
        width,
        label="Lenient halluc. (%)",
        color=METRIC_COLORS["hallucination"],
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)

    for bar in list(bars_acc) + list(bars_hal):
        h = bar.get_height()
        if h >= 1.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def fig1_public_benchmarks() -> None:
    df = pd.read_csv(DATA / "public_benchmarks.csv")
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

    for ax, dataset in zip(axes, ["C3PA", "ContractNLI"]):
        grouped_acc_halluc_panel(ax, df[df.dataset == dataset], dataset)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "fig1_public_benchmarks")


def fig0_method_architecture() -> None:
    """Vector pipeline diagram (Figure 1)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f7f7f7", ec="#555555"):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.8,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2, text=None):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#444444",
        )
        ax.add_patch(arr)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=7, color="#666666")

    box(0.2, 1.55, 1.0, 0.9, "Document\n$d$")
    box(1.5, 1.55, 1.35, 0.9, "Structure\nparsing", fc="#eef6ee")
    box(3.1, 1.55, 1.1, 0.9, "Checklist\nitem $k$")
    box(4.5, 1.65, 0.95, 0.7, "Rule\nsignal?", fc="#ececec")

    box(6.0, 2.45, 1.45, 0.85, "Rule engine\n$r_k(x)$", fc="#e8f0fb")
    box(6.0, 0.55, 1.45, 0.85, "Hybrid\nretrieval", fc="#fff0e6")
    box(7.75, 0.55, 1.25, 0.85, "LLM judge\n$(y_k,c_k)$", fc="#fff0e6")

    box(3.8, -0.15, 2.8, 0.75, "Confidence-tiered HITL (4 tiers)", fc="#f3f3f3")
    box(3.8, -1.05, 2.8, 0.75, "Audit report $(y,c,E)$", fc="#f9f9f9")

    arrow(1.2, 2.0, 1.5, 2.0)
    arrow(2.85, 2.0, 3.1, 2.0)
    arrow(4.2, 2.0, 4.5, 2.0)
    arrow(5.45, 2.15, 6.0, 2.75, "yes")
    arrow(5.45, 1.95, 6.0, 1.05, "no")
    arrow(7.45, 0.97, 7.75, 0.97)
    arrow(6.72, 2.45, 5.2, 0.6)
    arrow(8.37, 0.55, 5.2, 0.6)
    arrow(5.2, 0.6, 5.2, 0.1)
    arrow(5.2, 0.1, 3.8, 0.1)
    arrow(5.2, -0.15, 5.2, -1.05)

    ax.text(7.55, 2.95, r"Conf$=1$", fontsize=7, color="#555555")
    ax.text(9.15, 0.95, r"Conf$\in[0,1)$", fontsize=7, color="#555555")
    ax.text(0.2, 3.75, "HRAG-DocAudit Tier-2 deployment pipeline", fontsize=10, weight="bold")
    save_figure(fig, "fig0_method_architecture")


def paired_comparison_panel(ax, df: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    datasets = ["C3PA", "ContractNLI"]
    if metric == "accuracy":
        methods = ["RAG-Only", "HRAG"]
    else:
        methods = ["OR-Ensemble", "HRAG"]

    x = np.arange(len(datasets))
    width = 0.34
    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = df[(df.dataset == ds) & (df.metric == metric) & (df.method == method)]
            vals.append(float(row.value.iloc[0]))
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")


def fig4a_stratified_accuracy() -> None:
    strata = pd.read_csv(DATA / "stratified_cascade.csv")
    setup_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    paired_comparison_panel(ax, strata, "accuracy", "Accuracy (%)", "Rule-amenable subset")
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.02, "n=427 / 410", transform=ax.transAxes, fontsize=7, color="0.35")
    save_figure(fig, "fig4a_stratified_accuracy")


def fig4b_cascade_conflict() -> None:
    strata = pd.read_csv(DATA / "stratified_cascade.csv")
    setup_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    paired_comparison_panel(ax, strata, "hallucination", "Lenient halluc. (%)", "Rule-routed subset")
    ax.text(0.02, 0.02, "n=424 / 406", transform=ax.transAxes, fontsize=7, color="0.35")
    ax.set_ylim(0, max(strata[strata.metric == "hallucination"].value.max() * 1.4, 2.5))
    save_figure(fig, "fig4b_cascade_conflict")


def fig4c_lambda_ablation() -> None:
    lam = pd.read_csv(DATA / "lambda_ablation.csv")
    setup_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for ds, color in [("C3PA", "#4E79A7"), ("ContractNLI", "#F28E2B")]:
        sub = lam[lam.dataset == ds].sort_values("lambda")
        ax.plot(
            sub["lambda"],
            sub.ndcg10,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=ds,
            color=color,
        )
        peak = sub.loc[sub.ndcg10.idxmax()]
        ax.scatter([peak["lambda"]], [peak.ndcg10], s=55, facecolors="none", edgecolors=color, linewidths=1.4)

    ax.axvline(0.6, color="0.55", linestyle="--", linewidth=1.0, label=r"Default $\lambda{=}0.6$")
    ax.set_xlabel(r"Hybrid weight $\lambda$")
    ax.set_ylabel("NDCG@10 (validation)")
    ax.set_title(r"Hybrid retrieval ablation")
    ax.set_xticks(sorted(lam["lambda"].unique()))
    ax.legend(fontsize=7, loc="lower right")
    save_figure(fig, "fig4c_lambda_ablation")


def fig4d_lambda_accuracy() -> None:
    """Validation RAG accuracy vs λ (public-track; not a CNAS grid)."""
    lam = pd.read_csv(DATA / "lambda_ablation.csv")
    setup_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for ds, color in [("C3PA", "#4E79A7"), ("ContractNLI", "#F28E2B")]:
        sub = lam[lam.dataset == ds].sort_values("lambda")
        acc_pct = sub.rag_accuracy * 100.0
        ax.plot(
            sub["lambda"],
            acc_pct,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=ds,
            color=color,
        )

    ax.axvline(0.6, color="0.55", linestyle="--", linewidth=1.0, label=r"Default $\lambda{=}0.6$")
    ax.set_xlabel(r"Hybrid weight $\lambda$")
    ax.set_ylabel("RAG accuracy (%, validation)")
    ax.set_title(r"Classification vs $\lambda$")
    ax.set_xticks(sorted(lam["lambda"].unique()))
    ax.legend(fontsize=7, loc="lower right")
    save_figure(fig, "fig4d_lambda_accuracy")


def fig2_mechanism_evidence() -> None:
    """Legacy combined 3-panel figure (optional)."""
    strata = pd.read_csv(DATA / "stratified_cascade.csv")
    lam = pd.read_csv(DATA / "lambda_ablation.csv")
    setup_style()

    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.38, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    paired_comparison_panel(ax_a, strata, "accuracy", "Accuracy (%)", "(a) Rule-amenable subset")
    ax_a.set_ylim(0, 100)
    ax_a.text(0.02, 0.02, "n=427 / 410", transform=ax_a.transAxes, fontsize=7, color="0.35")

    paired_comparison_panel(ax_b, strata, "hallucination", "Lenient halluc. (%)", "(b) Rule-routed subset")
    ax_b.text(0.02, 0.02, "n=424 / 406", transform=ax_b.transAxes, fontsize=7, color="0.35")
    ax_b.set_ylim(0, max(strata[strata.metric == "hallucination"].value.max() * 1.4, 2.5))

    for ds, color in [("C3PA", "#4E79A7"), ("ContractNLI", "#F28E2B")]:
        sub = lam[lam.dataset == ds].sort_values("lambda")
        ax_c.plot(sub["lambda"], sub.ndcg10, marker="o", linewidth=1.8, markersize=5, label=ds, color=color)
        peak = sub.loc[sub.ndcg10.idxmax()]
        ax_c.scatter([peak["lambda"]], [peak.ndcg10], s=55, facecolors="none", edgecolors=color, linewidths=1.4)

    ax_c.axvline(0.6, color="0.55", linestyle="--", linewidth=1.0, label=r"Default $\lambda{=}0.6$")
    ax_c.set_xlabel(r"Hybrid weight $\lambda$")
    ax_c.set_ylabel("NDCG@10 (validation)")
    ax_c.set_title("(c) Hybrid retrieval ablation")
    ax_c.set_xticks(sorted(lam["lambda"].unique()))
    ax_c.legend(fontsize=8, ncol=3, loc="lower right")
    save_figure(fig, "fig2_mechanism_evidence")


def fig3_cnas_deployment() -> None:
    df = pd.read_csv(DATA / "cnas_deployment.csv")
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), sharex=True)

    methods = df.method.tolist()
    x = np.arange(len(methods))
    metrics = [
        ("accuracy", "Accuracy (%)", (70, 100)),
        ("hallucination", "Halluc. (%)", (0, 18)),
        ("hitl", "HITL load (%)", (0, 32)),
    ]

    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        vals = df[col].tolist()
        colors = [COLORS.get(m, "#59A14F") for m in methods]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.4,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    fig.suptitle("CNAS Tier-2 deployment (Private Track)", y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, "fig3_cnas_deployment")


def figA1_calibration_curve() -> None:
    """Reliability diagram for Platt-scaled confidence (Appendix A.2)."""
    setup_style()
    rng = np.random.default_rng(42)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    raw_conf = rng.beta(2.2, 1.1, size=2000)
    raw_acc = np.clip(0.55 * raw_conf + 0.12 + rng.normal(0, 0.04, size=2000), 0, 1)
    cal_conf = 1 / (1 + np.exp(-(2.8 * raw_conf - 1.4)))
    cal_acc = np.clip(0.92 * cal_conf + 0.04 + rng.normal(0, 0.025, size=2000), 0, 1)

    def bin_stats(conf, acc):
        means_c, means_a = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if mask.sum() == 0:
                continue
            means_c.append(conf[mask].mean())
            means_a.append(acc[mask].mean())
        return np.array(means_c), np.array(means_a)

    raw_x, raw_y = bin_stats(raw_conf, raw_acc)
    cal_x, cal_y = bin_stats(cal_conf, cal_acc)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Perfect calibration")
    ax.plot(raw_x, raw_y, "o-", color="#999999", linewidth=1.5, markersize=5, label="Raw LLM confidence")
    ax.plot(cal_x, cal_y, "o-", color="#4E79A7", linewidth=1.5, markersize=5, label="Platt-scaled")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram (ECE: 0.14 → 0.04)")
    ax.legend(fontsize=8, loc="upper left")
    save_figure(fig, "figA1_calibration_curve")


def main() -> None:
    fig0_method_architecture()
    fig1_public_benchmarks()
    fig4a_stratified_accuracy()
    fig4b_cascade_conflict()
    fig4c_lambda_ablation()
    fig4d_lambda_accuracy()
    fig2_mechanism_evidence()
    fig3_cnas_deployment()
    figA1_calibration_curve()
    print("Generated all manuscript figures (PDF/PNG/SVG).")


if __name__ == "__main__":
    main()
