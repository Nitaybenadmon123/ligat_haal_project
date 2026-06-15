"""Poster chart: Competitiveness Indicators 2020/21 vs 2006/07 (large fonts)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "data" / "processed" / "title_race_analysis" / "season_summary_statistics.csv"
OUT_DIR = ROOT / "notebooks" / "outputs" / "research_summary" / "charts"
OUT_PNG = OUT_DIR / "competitiveness_indicators_2020_21_vs_2006_07_poster.png"
OUT_CSV = OUT_DIR / "competitiveness_indicators_2020_21_vs_2006_07_poster.csv"

METRICS = [
    ("avg_gap", "Avg gap"),
    ("final_gap", "Final gap"),
    ("dominant_pct", "Dominant rounds %"),
    ("leadership_changes", "Leadership changes"),
    ("distinct_leaders", "Different leaders"),
    ("tight_pct", "Tight rounds %"),
]

SEASONS = ("2006/07", "2020/21")
COLORS = {"2006/07": "#4C78A8", "2020/21": "#F58518"}


def main() -> None:
    df = pd.read_csv(DATA)
    rows = []
    for col, label in METRICS:
        row = {"metric": label}
        for season in SEASONS:
            val = df.loc[df["season"] == season, col].iloc[0]
            row[season] = round(float(val), 1) if col != "leadership_changes" and col != "distinct_leaders" else int(val)
        rows.append(row)
    plot_df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    labels = plot_df["metric"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 9))
    for i, season in enumerate(SEASONS):
        vals = plot_df[season].tolist()
        offset = -width / 2 if i == 0 else width / 2
        bars = ax.bar(x + offset, vals, width, label=season, color=COLORS[season], edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{val:g}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_title(
        "Competitiveness Indicators: 2020/21 vs 2006/07",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylabel("Metric value", fontsize=18, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=15,
        fontweight="bold",
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, max(plot_df[SEASONS[0]].max(), plot_df[SEASONS[1]].max()) * 1.18)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.35)
    ax.text(
        0.02,
        0.98,
        "Lower is better for gaps/dominance  |  Higher is better for changes/leaders/tight rounds",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Data:  {OUT_CSV}")


if __name__ == "__main__":
    main()
