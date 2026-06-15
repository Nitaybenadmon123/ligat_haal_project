"""Poster radar chart: Season Quality Index 2020/21 vs 2006/07 (large fonts)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "data" / "processed" / "title_race_analysis" / "season_summary_statistics.csv"
OUT_DIR = ROOT / "notebooks" / "outputs" / "research_summary" / "charts"
OUT_PNG = OUT_DIR / "season_quality_index_radar_2020_21_vs_2006_07_poster.png"
OUT_CSV = OUT_DIR / "season_quality_index_radar_2020_21_vs_2006_07_poster.csv"

SEASONS = ("2006/07", "2020/21")
COLORS = {"2006/07": "#4C78A8", "2020/21": "#F58518"}

DIMENSIONS = [
    ("distinct_leaders", "Different leaders", "high"),
    ("leadership_changes", "Leadership changes", "high"),
    ("dominant_pct", "Low dominance %", "low"),
    ("final_gap", "Low final gap", "low"),
    ("avg_gap", "Low avg gap", "low"),
    ("tight_pct", "Tight rounds %", "raw"),
]


def _normalize(df: pd.DataFrame, col: str, val: float, direction: str) -> float:
    if direction == "raw":
        return float(val)
    lo, hi = df[col].min(), df[col].max()
    if hi == lo:
        return 50.0
    if direction == "high":
        return 100.0 * (val - lo) / (hi - lo)
    return 100.0 * (hi - val) / (hi - lo)


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        row = df.loc[df["season"] == season].iloc[0]
        scores: dict[str, float] = {"season": season}
        for col, label, direction in DIMENSIONS:
            scores[label] = round(_normalize(df, col, row[col], direction), 1)
        dim_labels = [d[1] for d in DIMENSIONS]
        scores["overall_score"] = round(sum(scores[l] for l in dim_labels) / len(dim_labels), 1)
        rows.append(scores)
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA)
    scores = build_scores(df)
    labels = [d[1] for d in DIMENSIONS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for season in SEASONS:
        srow = scores.loc[scores["season"] == season].iloc[0]
        values = [srow[label] for label in labels]
        values_closed = values + [values[0]]
        angles_closed = np.concatenate([angles, [angles[0]]])
        ax.plot(
            angles_closed,
            values_closed,
            color=COLORS[season],
            linewidth=3.5,
            label=f"{season} | Score: {srow['overall_score']}",
        )
        ax.fill(angles_closed, values_closed, color=COLORS[season], alpha=0.18)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=22, fontweight="bold")
    ax.set_rlabel_position(180 / n)
    ax.grid(color="#bbbbbb", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=26, fontweight="bold")
    ax.tick_params(axis="x", pad=48)

    ax.set_title(
        "Season Quality Index: 2020/21 vs 2006/07\nMulti-dimensional competitiveness quality score",
        fontsize=34,
        fontweight="bold",
        pad=52,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.14), fontsize=24, framealpha=0.95)

    fig.subplots_adjust(left=0.06, right=0.82, top=0.86, bottom=0.06)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Data:  {OUT_CSV}")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
