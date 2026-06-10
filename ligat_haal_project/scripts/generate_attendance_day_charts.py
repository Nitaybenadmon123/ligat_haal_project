"""
Generate attendance-by-weekday analysis charts from match-level Ligat Ha'al data.

Run:
    python scripts/generate_attendance_day_charts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants (aligned with build_research_summary_table.py)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
DATA = NOTEBOOKS / "data"
ATTENDANCE_DIR = NOTEBOOKS / "attendance"
OUTPUT = NOTEBOOKS / "outputs" / "research_summary"
CHART_DIR = OUTPUT / "charts" / "attendance_day_analysis"

ATTENDANCE_MIN_COVERAGE_PCT = 60
MIN_MATCHES_PER_DAY_RELIABLE = 30

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

sys.path.insert(0, str(ROOT / "scripts"))
try:
    from build_research_summary_table import (  # type: ignore
        season_sort_key,
        list_seasons_from_matches,
    )
except ImportError:
    def season_sort_key(season: str) -> tuple[int, int]:
        y1, y2 = season.split("/")
        return int(y1), int(y2)

    def list_seasons_from_matches(matches: pd.DataFrame) -> list[str]:
        seasons = sorted(matches["season"].dropna().unique(), key=season_sort_key)
        return [s for s in seasons if season_sort_key(s) >= (2006, 7)]


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------
def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in lower_map:
            return lower_map[key]
    return None


def _slug_to_season(slug: str) -> str | None:
    if slug == "2024_2025":
        return "2024/25"
    m = re.match(r"^(\d{4})_(\d{2})$", slug)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def _season_to_slug(season: str) -> str:
    return season.replace("/", "_")


def _regular_attendance_paths(season: str) -> list[Path]:
    slug = _season_to_slug(season)
    paths = [ATTENDANCE_DIR / f"ligat_haal_{slug}_attendance.csv"]
    if season == "2024/25":
        paths.insert(0, ATTENDANCE_DIR / "ligat_haal_2024_2025_attendance.csv")
    return paths


def _clean_attendance_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(series, errors="coerce")


def _normalize_attendance_file(
    df: pd.DataFrame,
    source_file: Path,
    competition_stage: str,
    season_fallback: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    season_col = _pick_column(out, ["Season", "season"])
    date_col = _pick_column(out, ["Date", "date", "MatchDate", "match_date"])
    att_col = _pick_column(out, ["Attendance", "attendance"])
    home_col = _pick_column(out, ["HomeTeam", "home_team", "home", "Home"])
    away_col = _pick_column(out, ["AwayTeam", "away_team", "away", "Away"])
    stage_col = _pick_column(out, ["Stage", "stage", "competition_stage"])

    if att_col is None or date_col is None:
        return pd.DataFrame()

    out["season"] = out[season_col] if season_col else season_fallback
    out["date_raw"] = out[date_col]
    out["attendance"] = _clean_attendance_series(out[att_col])
    out["home_team"] = out[home_col] if home_col else np.nan
    out["away_team"] = out[away_col] if away_col else np.nan
    out["competition_stage"] = (
        out[stage_col] if stage_col else competition_stage
    )
    out["source_file"] = source_file.name

    out["match_date"] = pd.to_datetime(out["date_raw"], format="%d/%m/%y", errors="coerce")
    if out["match_date"].isna().all():
        out["match_date"] = pd.to_datetime(out["date_raw"], errors="coerce")

    out["day_of_week"] = out["match_date"].dt.dayofweek.map(
        lambda d: WEEKDAY_ORDER[int(d)] if pd.notna(d) and 0 <= int(d) <= 6 else np.nan
    )
    out["is_saturday"] = out["match_date"].dt.dayofweek == 5
    return out


def load_season_attendance_raw(season: str) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for path in _regular_attendance_paths(season):
        if path.exists():
            norm = _normalize_attendance_file(
                pd.read_csv(path), path, "regular", season_fallback=season
            )
            if not norm.empty:
                frames.append(norm)
            break

    playoff_path = ATTENDANCE_DIR / f"ligat_haal_{_season_to_slug(season)}_playoffs_attendance.csv"
    if playoff_path.exists():
        norm = _normalize_attendance_file(
            pd.read_csv(playoff_path), playoff_path, "playoff", season_fallback=season
        )
        if not norm.empty:
            frames.append(norm)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def discover_seasons() -> list[str]:
    matches_path = DATA / "matches" / "matches_all_seasons_ligat_haal_transfermarkt_dated.csv"
    if matches_path.exists():
        matches = pd.read_csv(matches_path)
        return list_seasons_from_matches(matches)

    seasons: set[str] = set()
    for path in ATTENDANCE_DIR.glob("ligat_haal_*_attendance.csv"):
        if "playoffs" in path.name or "2004_2024" in path.name:
            continue
        slug = path.stem.replace("ligat_haal_", "").replace("_attendance", "")
        s = _slug_to_season(slug)
        if s:
            seasons.add(s)
    return sorted(seasons, key=season_sort_key)


def season_coverage_pct(raw: pd.DataFrame) -> float:
    total = len(raw)
    if total == 0:
        return 0.0
    valid = (raw["attendance"] > 0).sum()
    return 100.0 * valid / total


def load_analysis_pool() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (valid_matches, season_coverage_df, included_seasons)."""
    coverage_rows: list[dict] = []
    valid_frames: list[pd.DataFrame] = []
    included: list[str] = []

    for season in discover_seasons():
        raw = load_season_attendance_raw(season)
        if raw is None or raw.empty:
            coverage_rows.append({
                "season": season,
                "total_matches": 0,
                "valid_matches": 0,
                "coverage_pct": 0.0,
                "included": False,
            })
            continue

        cov = season_coverage_pct(raw)
        valid_n = int((raw["attendance"] > 0).sum())
        included_flag = cov >= ATTENDANCE_MIN_COVERAGE_PCT
        coverage_rows.append({
            "season": season,
            "total_matches": len(raw),
            "valid_matches": valid_n,
            "coverage_pct": cov,
            "included": included_flag,
        })
        if not included_flag:
            continue

        valid = raw.loc[
            (raw["attendance"] > 0)
            & raw["match_date"].notna()
            & raw["day_of_week"].notna()
        ].copy()
        if not valid.empty:
            valid_frames.append(valid)
            included.append(season)

    pool = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows)
    return pool, coverage_df, included


def _weekday_stats(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day in WEEKDAY_ORDER:
        sub = pool.loc[pool["day_of_week"] == day, "attendance"]
        if sub.empty:
            continue
        rows.append({
            "day_of_week": day,
            "matches": len(sub),
            "mean_attendance": sub.mean(),
            "median_attendance": sub.median(),
            "min_attendance": sub.min(),
            "max_attendance": sub.max(),
            "std_attendance": sub.std(ddof=1) if len(sub) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def _low_sample_label(n: int) -> str:
    return "low sample" if n < MIN_MATCHES_PER_DAY_RELIABLE else ""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_01_boxplot(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    stats = _weekday_stats(pool)
    stats.to_csv(out_dir / "01_attendance_boxplot_by_weekday_summary.csv", index=False, encoding="utf-8-sig")
    created.append("01_attendance_boxplot_by_weekday_summary.csv")

    data = [pool.loc[pool["day_of_week"] == d, "attendance"].values for d in WEEKDAY_ORDER]
    labels = []
    for d in WEEKDAY_ORDER:
        n = len(pool.loc[pool["day_of_week"] == d])
        tag = _low_sample_label(n)
        labels.append(f"{d}\nn={n}" + (f"\n({tag})" if tag else ""))

    fig, ax = plt.subplots(figsize=(12, 7))
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.85)
    ax.set_title("Match Attendance Distribution by Weekday (Boxplot)")
    ax.set_ylabel("Attendance")
    ax.set_xlabel("Day of week")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "01_attendance_boxplot_by_weekday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("01_attendance_boxplot_by_weekday.png")
    return created


def chart_02_mean_median(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    stats = _weekday_stats(pool)
    stats["low_sample"] = stats["matches"] < MIN_MATCHES_PER_DAY_RELIABLE
    stats.to_csv(out_dir / "02_attendance_mean_vs_median_by_weekday_summary.csv", index=False, encoding="utf-8-sig")
    created.append("02_attendance_mean_vs_median_by_weekday_summary.csv")

    x = np.arange(len(stats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - w / 2, stats["mean_attendance"], w, label="Mean", color="#4c78a8")
    ax.bar(x + w / 2, stats["median_attendance"], w, label="Median", color="#f58518")
    ax.set_xticks(x)
    labels = []
    for _, row in stats.iterrows():
        tag = " (low sample)" if row["low_sample"] else ""
        labels.append(f"{row['day_of_week']}\nn={int(row['matches'])}{tag}")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Attendance")
    ax.set_title("Mean vs Median Attendance by Weekday")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "02_attendance_mean_vs_median_by_weekday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("02_attendance_mean_vs_median_by_weekday.png")
    return created


def chart_03_scatter(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    export_cols = [
        "season", "date_raw", "day_of_week", "attendance",
        "competition_stage", "source_file", "home_team", "away_team",
    ]
    export = pool[[c for c in export_cols if c in pool.columns]].rename(columns={"date_raw": "date"})
    export.to_csv(out_dir / "03_attendance_match_scatter_by_weekday_data.csv", index=False, encoding="utf-8-sig")
    created.append("03_attendance_match_scatter_by_weekday_data.csv")

    day_to_idx = {d: i for i, d in enumerate(WEEKDAY_ORDER)}
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(12, 7))
    for day in WEEKDAY_ORDER:
        sub = pool.loc[pool["day_of_week"] == day]
        if sub.empty:
            continue
        idx = day_to_idx[day]
        jitter = rng.uniform(-0.22, 0.22, size=len(sub))
        ax.scatter(idx + jitter, sub["attendance"], alpha=0.35, s=18, color="#4c78a8")
        mean_val = sub["attendance"].mean()
        ax.hlines(mean_val, idx - 0.35, idx + 0.35, colors="#e45756", linewidth=2.5, label="_nolegend_")

    means = pool.groupby("day_of_week")["attendance"].mean().reindex(WEEKDAY_ORDER)
    ax.plot(range(len(WEEKDAY_ORDER)), means.values, color="#e45756", linewidth=1.5, marker="D",
            markersize=6, label="Mean per weekday")

    labels = []
    for d in WEEKDAY_ORDER:
        n = len(pool.loc[pool["day_of_week"] == d])
        tag = _low_sample_label(n)
        labels.append(f"{d}\nn={n}" + (f"\n({tag})" if tag else ""))
    ax.set_xticks(range(len(WEEKDAY_ORDER)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Attendance")
    ax.set_title("Match-Level Attendance by Weekday (Scatter + Mean)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "03_attendance_match_scatter_by_weekday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("03_attendance_match_scatter_by_weekday.png")
    return created


def chart_04_reliable_days(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    stats = _weekday_stats(pool)
    reliable = stats[stats["matches"] >= MIN_MATCHES_PER_DAY_RELIABLE].copy()
    reliable.to_csv(out_dir / "04_attendance_reliable_days_only_summary.csv", index=False, encoding="utf-8-sig")
    created.append("04_attendance_reliable_days_only_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    if reliable.empty:
        ax.text(0.5, 0.5, "No weekdays meet the reliability threshold.", ha="center", va="center")
    else:
        x = np.arange(len(reliable))
        bars = ax.bar(x, reliable["mean_attendance"], color="#72b7b2", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['day_of_week']}\nn={int(r['matches'])}" for _, r in reliable.iterrows()],
                           rotation=20, ha="right")
        for bar, val in zip(bars, reliable["mean_attendance"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average attendance")
    ax.set_title(
        "Average Attendance by Weekday (Reliable Days Only)\n"
        f"Days with n < {MIN_MATCHES_PER_DAY_RELIABLE} were excluded from the reliable ranking."
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "04_attendance_reliable_days_only.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("04_attendance_reliable_days_only.png")
    return created


def chart_05_share_vs_match(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    total_matches = len(pool)
    total_att = pool["attendance"].sum()
    rows = []
    for day in WEEKDAY_ORDER:
        sub = pool.loc[pool["day_of_week"] == day]
        if sub.empty:
            continue
        m_share = 100.0 * len(sub) / total_matches
        a_share = 100.0 * sub["attendance"].sum() / total_att
        eff = a_share / m_share if m_share > 0 else np.nan
        rows.append({
            "day_of_week": day,
            "matches": len(sub),
            "match_share_pct": m_share,
            "total_attendance": sub["attendance"].sum(),
            "attendance_share_pct": a_share,
            "efficiency_ratio": eff,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "05_attendance_share_vs_match_share_by_weekday_summary.csv", index=False, encoding="utf-8-sig")
    created.append("05_attendance_share_vs_match_share_by_weekday_summary.csv")

    x = np.arange(len(summary))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - w / 2, summary["match_share_pct"], w, label="% of matches", color="#4c78a8")
    ax.bar(x + w / 2, summary["attendance_share_pct"], w, label="% of total attendance", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['day_of_week']}\nn={int(r['matches'])}" for _, r in summary.iterrows()],
        rotation=20, ha="right",
    )
    ax.set_ylabel("Percentage")
    ax.set_title("Match Share vs Attendance Share by Weekday")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "05_attendance_share_vs_match_share_by_weekday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("05_attendance_share_vs_match_share_by_weekday.png")
    return created


def chart_06_saturday_trend(pool: pd.DataFrame, included_seasons: list[str], out_dir: Path) -> list[str]:
    created: list[str] = []
    rows = []
    for season in sorted(included_seasons, key=season_sort_key):
        sub = pool.loc[pool["season"] == season]
        sat = sub.loc[sub["is_saturday"], "attendance"]
        wkd = sub.loc[~sub["is_saturday"], "attendance"]
        sat_avg = float(sat.mean()) if not sat.empty else np.nan
        wkd_avg = float(wkd.mean()) if not wkd.empty else np.nan
        rows.append({
            "season": season,
            "saturday_matches": len(sat),
            "weekday_matches": len(wkd),
            "saturday_avg_attendance": sat_avg,
            "weekday_avg_attendance": wkd_avg,
            "saturday_minus_weekday": sat_avg - wkd_avg if pd.notna(sat_avg) and pd.notna(wkd_avg) else np.nan,
        })
    trend = pd.DataFrame(rows)
    trend.to_csv(out_dir / "06_saturday_vs_weekday_attendance_trend_summary.csv", index=False, encoding="utf-8-sig")
    created.append("06_saturday_vs_weekday_attendance_trend_summary.csv")

    seasons = trend["season"].tolist()
    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, trend["saturday_avg_attendance"], marker="o", linewidth=2, label="Saturday avg")
    ax.plot(x, trend["weekday_avg_attendance"], marker="s", linewidth=2, label="Non-Saturday avg")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_ylabel("Average attendance")
    ax.set_title(
        f"Saturday vs Non-Saturday Attendance by Season\n"
        f"(seasons with coverage >= {ATTENDANCE_MIN_COVERAGE_PCT}%)"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "06_saturday_vs_weekday_attendance_trend.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("06_saturday_vs_weekday_attendance_trend.png")
    return created


def chart_07_mean_ci(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    rows = []
    for day in WEEKDAY_ORDER:
        sub = pool.loc[pool["day_of_week"] == day, "attendance"]
        if sub.empty:
            continue
        n = len(sub)
        mean = float(sub.mean())
        std = float(sub.std(ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        ci_half = 1.96 * se if n >= 2 else 0.0
        rows.append({
            "day_of_week": day,
            "matches": n,
            "mean_attendance": mean,
            "std_attendance": std,
            "standard_error": se,
            "ci95_low": mean - ci_half,
            "ci95_high": mean + ci_half,
            "low_sample": n < MIN_MATCHES_PER_DAY_RELIABLE,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "07_attendance_mean_ci_by_weekday_summary.csv", index=False, encoding="utf-8-sig")
    created.append("07_attendance_mean_ci_by_weekday_summary.csv")

    x = np.arange(len(summary))
    colors = ["#b279a2" if ls else "#4c78a8" for ls in summary["low_sample"]]
    yerr = 1.96 * summary["standard_error"].where(summary["matches"] >= 2, 0.0)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x, summary["mean_attendance"], color=colors, alpha=0.9, yerr=yerr, capsize=4, ecolor="#333333")
    ax.set_xticks(x)
    labels = []
    for _, row in summary.iterrows():
        tag = " (low sample)" if row["low_sample"] else ""
        labels.append(f"{row['day_of_week']}\nn={int(row['matches'])}{tag}")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean attendance")
    ax.set_title("Mean Attendance by Weekday with 95% Confidence Intervals")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "07_attendance_mean_ci_by_weekday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("07_attendance_mean_ci_by_weekday.png")
    return created


def chart_08_top10(pool: pd.DataFrame, out_dir: Path) -> list[str]:
    created: list[str] = []
    top10 = pool.nlargest(10, "attendance").copy()
    export_cols = [
        "season", "date_raw", "day_of_week", "home_team", "away_team",
        "attendance", "source_file", "competition_stage",
    ]
    top10[[c for c in export_cols if c in top10.columns]].rename(columns={"date_raw": "date"}).to_csv(
        out_dir / "08_top_10_attendance_matches.csv", index=False, encoding="utf-8-sig"
    )
    created.append("08_top_10_attendance_matches.csv")

    counts = top10["day_of_week"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(WEEKDAY_ORDER)), counts.values, color="#54a24b", alpha=0.9)
    ax.set_xticks(range(len(WEEKDAY_ORDER)))
    ax.set_xticklabels(WEEKDAY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Number of top-10 matches")
    ax.set_title("Weekday Distribution of Top 10 Highest-Attendance Matches")
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val, str(int(val)), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "08_top_attendance_matches_by_weekday.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    created.append("08_top_attendance_matches_by_weekday.png")
    return created


def write_readme(out_dir: Path, pool: pd.DataFrame, included: list[str]) -> str:
    text = f"""Attendance Day-of-Week Analysis
=============================

Data rules
----------
1. Valid attendance: Attendance > 0
2. Season inclusion: coverage >= {ATTENDANCE_MIN_COVERAGE_PCT}% of matches in that season's combined regular+playoff files
3. Low-sample flag: weekdays with fewer than {MIN_MATCHES_PER_DAY_RELIABLE} valid matches are marked as low sample
4. Dates parsed from match files; weekday names in English (Monday-Sunday)
5. Regular season and playoff matches are combined per season

Included seasons ({len(included)}): {', '.join(sorted(included, key=season_sort_key))}
Valid matches in analysis pool: {len(pool)}

Chart guide
-----------
01_attendance_boxplot_by_weekday.png
    Distribution, median, and outliers per weekday. Use to see whether Friday's high average is driven by outliers.

02_attendance_mean_vs_median_by_weekday.png
    Compares mean and median. Large gaps suggest outlier influence.

03_attendance_match_scatter_by_weekday.png
    Every match as a point with jitter; red markers/lines show weekday means. Shows Friday's tiny sample visually.

04_attendance_reliable_days_only.png
    Average attendance only for weekdays with n >= {MIN_MATCHES_PER_DAY_RELIABLE}. Low-sample days excluded from ranking.

05_attendance_share_vs_match_share_by_weekday.png
    Compares share of matches vs share of total spectators. Efficiency ratio > 1 means overperformance per match volume.

06_saturday_vs_weekday_attendance_trend.png
    Season-by-season Saturday average vs non-Saturday average for included seasons.

07_attendance_mean_ci_by_weekday.png
    Mean attendance with 95% CI (1.96 * SE). Purple bars = low sample; Friday CI is wide/unreliable due to n=7.

08_top_attendance_matches_by_weekday.png
    Which weekdays host the top 10 single-match attendance records.

Important note on Friday
------------------------
Friday shows a very high average attendance in the full sample, but typically has only a handful of matches (low sample).
Treat Friday as a potential scheduling opportunity, not a confirmed league-wide trend, until more Friday games are observed.
"""
    path = out_dir / "README_attendance_day_analysis.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    pool, coverage_df, included = load_analysis_pool()
    coverage_df.to_csv(CHART_DIR / "00_season_coverage_summary.csv", index=False, encoding="utf-8-sig")

    if pool.empty:
        print("No valid attendance matches found for seasons meeting coverage threshold.")
        return

    generated: list[str] = ["00_season_coverage_summary.csv"]
    generated.extend(chart_01_boxplot(pool, CHART_DIR))
    generated.extend(chart_02_mean_median(pool, CHART_DIR))
    generated.extend(chart_03_scatter(pool, CHART_DIR))
    generated.extend(chart_04_reliable_days(pool, CHART_DIR))
    generated.extend(chart_05_share_vs_match(pool, CHART_DIR))
    generated.extend(chart_06_saturday_trend(pool, included, CHART_DIR))
    generated.extend(chart_07_mean_ci(pool, CHART_DIR))
    generated.extend(chart_08_top10(pool, CHART_DIR))
    generated.append(Path(write_readme(CHART_DIR, pool, included)).name)

    print("=" * 60)
    print("ATTENDANCE DAY-OF-WEEK ANALYSIS")
    print("=" * 60)
    print(f"Output folder: {CHART_DIR}")
    print(f"Seasons included: {len(included)}")
    print(f"Valid matches: {len(pool)}")
    print("\nGenerated files:")
    for fname in generated:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
