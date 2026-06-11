"""
Advanced season-level competitiveness features for build_research_summary_table.py.

Parts: offensive concentration, rank dynamics, combined competitiveness,
playoff impact comparison, data quality, charts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "data"
NOTEBOOKS = ROOT / "notebooks"
OUTPUT = NOTEBOOKS / "outputs" / "research_summary"
ADV_CHARTS = OUTPUT / "charts" / "advanced_competitiveness_features"

RANK_CHANGES_PATH = DATA / "processed" / "rank_dynamics" / "rank_changes_export.csv"
BUDGET_PATH = DATA / "demographic" / "ligat_haal_demographics_budget_by_season.csv"

ROUND_PAIR_RE = re.compile(
    r"after(?: mini-league)? round (\d+)\s+vs\s+after(?: mini-league)? round (\d+)",
    re.IGNORECASE,
)
FAKE_CLUB_PATTERNS = ("for 2 clubs", "for 3 clubs", "2 clubs", "3 clubs")

sys.path.insert(0, str(ROOT / "scripts"))
try:
    from player_goal_contributions import (  # type: ignore
        assists_path_for_season,
        is_valid_stats_file,
        scorers_path_for_season,
    )
except ImportError:
    is_valid_stats_file = None  # type: ignore
    scorers_path_for_season = None  # type: ignore
    assists_path_for_season = None  # type: ignore

# Known seasons without Transfermarkt assist stats
ASSISTS_UNAVAILABLE_SEASONS = {"2007/08", "2008/09"}


def _gini(values: np.ndarray) -> float:
    arr = np.sort(values.astype(float))
    n = len(arr)
    if n == 0 or np.nansum(arr) == 0:
        return np.nan
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def _minmax_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        out = pd.Series(50.0, index=series.index)
        out[series.isna()] = np.nan
        return out
    norm = (series - lo) / (hi - lo) * 100
    if not higher_is_better:
        norm = 100 - norm
    return norm


def _is_fake_club(club: Any) -> bool:
    if pd.isna(club):
        return True
    s = str(club).strip().lower()
    return any(p in s for p in FAKE_CLUB_PATTERNS)


def _valid_goals_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 40:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or "goals" not in df.columns:
        return False
    vals = pd.to_numeric(df["goals"], errors="coerce")
    if vals.notna().sum() == 0:
        return False
    return bool(vals.max() <= 45)


def _load_scorers_season(season: str) -> pd.DataFrame | None:
    if scorers_path_for_season is None:
        return None
    path = scorers_path_for_season(season)
    if not _valid_goals_file(path):
        return None
    df = pd.read_csv(path)
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce")
    df = df[df["goals"].notna() & (df["goals"] >= 0) & (df["goals"] <= 45)]
    return df if not df.empty else None


def compute_offensive_concentration(
    seasons: list[str],
    total_goals_by_season: dict[str, float | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (season summary DataFrame, audit DataFrame)."""
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for season in seasons:
        base = {
            "season": season,
            "top_scorer_goals": np.nan,
            "share_of_league_goals_by_top_3_scorers": np.nan,
            "offensive_concentration_gini": np.nan,
            "top_3_scorers_goals_total": np.nan,
            "top_3_scorers_names": "",
            "player_stats_goals_available": False,
        }
        sc = _load_scorers_season(season)
        if sc is None:
            rows.append(base)
            continue

        base["player_stats_goals_available"] = True
        # Aggregate goals per player (sum across clubs if listed separately)
        agg = (
            sc.groupby(["player", "player_url"], as_index=False)
            .agg(goals=("goals", "sum"), club=("club", "first"))
            .sort_values("goals", ascending=False)
        )
        if agg.empty:
            rows.append(base)
            continue

        top3 = agg.head(3)
        top_scorer_goals = int(top3.iloc[0]["goals"])
        top3_total = int(top3["goals"].sum())
        names = []
        for _, r in top3.iterrows():
            club = str(r["club"]) if not _is_fake_club(r["club"]) else str(r["club"])
            names.append(f"{r['player']} ({club}, {int(r['goals'])})")
        gini_val = _gini(agg["goals"].values)

        league_goals = total_goals_by_season.get(season)
        share = np.nan
        if league_goals and league_goals > 0:
            share = round(100 * top3_total / league_goals, 2)

        base.update(
            {
                "top_scorer_goals": top_scorer_goals,
                "share_of_league_goals_by_top_3_scorers": share,
                "offensive_concentration_gini": round(gini_val, 4) if pd.notna(gini_val) else np.nan,
                "top_3_scorers_goals_total": top3_total,
                "top_3_scorers_names": "; ".join(names),
            }
        )
        rows.append(base)

        top3_urls = set(top3["player_url"].tolist())
        rank = 1
        for _, r in agg.iterrows():
            audit_rows.append(
                {
                    "season": season,
                    "rank": rank,
                    "player": r["player"],
                    "club": r["club"],
                    "goals": int(r["goals"]),
                    "source_file": scorers_path_for_season(season).name if scorers_path_for_season else "",
                    "included_in_top_3": r["player_url"] in top3_urls,
                }
            )
            rank += 1

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def _parse_round_pair(label: str) -> tuple[int | None, int | None]:
    m = ROUND_PAIR_RE.search(str(label))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _stage_sort_key(stage: str) -> int:
    s = str(stage).lower()
    if "regular" in s:
        return 0
    if "championship" in s:
        return 1
    if "relegation" in s:
        return 2
    return 3


def compute_rank_dynamics(seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rank volatility metrics from rank_changes_export.csv."""
    empty_summary = pd.DataFrame(
        {
            "season": seasons,
            "avg_rank_changes_per_round": np.nan,
            "avg_rank_position_movement_per_round": np.nan,
            "max_single_round_rank_swing": np.nan,
            "late_season_rank_volatility": np.nan,
            "title_zone_rank_volatility": np.nan,
            "relegation_zone_rank_volatility": np.nan,
            "relegation_vs_title_volatility_ratio": np.nan,
            "rank_dynamics_data_available": False,
        }
    )
    if not RANK_CHANGES_PATH.exists():
        return empty_summary, pd.DataFrame()

    raw = pd.read_csv(RANK_CHANGES_PATH)
    raw["team"] = raw["team"].astype(str)
    raw["prev_rank"] = pd.to_numeric(raw["prev_rank"], errors="coerce")
    raw["new_rank"] = pd.to_numeric(raw["new_rank"], errors="coerce")
    raw["abs_change"] = (raw["new_rank"] - raw["prev_rank"]).abs()
    raw["round_to"], raw["round_from"] = zip(
        *raw["segment_label"].map(lambda x: _parse_round_pair(x))
    )
    raw["stage_sort"] = raw["stage"].map(_stage_sort_key)

    audit_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for season in seasons:
        sub = raw[raw["season"] == season].copy()
        if sub.empty:
            summary_rows.append({"season": season, "rank_dynamics_data_available": False})
            continue

        sub_valid = sub[sub["round_to"].notna()].copy()
        if sub_valid.empty:
            summary_rows.append({"season": season, "rank_dynamics_data_available": False})
            continue

        # Per transition metrics
        trans = (
            sub_valid.groupby(["stage", "segment_label", "round_from", "round_to", "stage_sort"], as_index=False)
            .agg(
                teams_changed=("abs_change", lambda s: int((s > 0).sum())),
                avg_movement=("abs_change", "mean"),
                max_swing=("abs_change", "max"),
            )
        )

        avg_changes = float(trans["teams_changed"].mean())
        avg_movement = float(trans["avg_movement"].mean())
        max_swing = float(trans["max_swing"].max())

        # Late season: last 5 transitions by stage order then round_to
        trans_sorted = trans.sort_values(["stage_sort", "round_to"])
        late_trans = trans_sorted.tail(5)
        late_rounds_note = len(late_trans)
        late_vol = float(late_trans["avg_movement"].mean()) if not late_trans.empty else np.nan

        # Teams in zones at start of late window (prev_rank from first late transition)
        late_labels = set(late_trans["segment_label"].tolist())
        late_rows = sub_valid[sub_valid["segment_label"].isin(late_labels)]
        if late_rows.empty:
            late_rows = sub_valid

        # Title zone: prev_rank <= 4 in regular/championship; all in relegation playoff excluded
        title_rows = late_rows[
            late_rows["stage"].astype(str).str.contains("regular|championship", case=False, regex=True)
            & (late_rows["prev_rank"] <= 4)
        ]
        rel_rows = late_rows[
            late_rows["stage"].astype(str).str.contains("regular|relegation", case=False, regex=True)
        ]
        if not rel_rows.empty:
            max_rank = rel_rows.groupby("segment_label")["prev_rank"].max().max()
            if pd.notna(max_rank):
                rel_rows = rel_rows[rel_rows["prev_rank"] >= max_rank - 3]

        title_vol = float(title_rows["abs_change"].mean()) if not title_rows.empty else np.nan
        rel_vol = float(rel_rows["abs_change"].mean()) if not rel_rows.empty else np.nan
        ratio = np.nan
        if pd.notna(title_vol) and title_vol > 0 and pd.notna(rel_vol):
            ratio = round(rel_vol / title_vol, 3)

        summary_rows.append(
            {
                "season": season,
                "avg_rank_changes_per_round": round(avg_changes, 2),
                "avg_rank_position_movement_per_round": round(avg_movement, 3),
                "max_single_round_rank_swing": int(max_swing) if pd.notna(max_swing) else np.nan,
                "late_season_rank_volatility": round(late_vol, 3) if pd.notna(late_vol) else np.nan,
                "title_zone_rank_volatility": round(title_vol, 3) if pd.notna(title_vol) else np.nan,
                "relegation_zone_rank_volatility": round(rel_vol, 3) if pd.notna(rel_vol) else np.nan,
                "relegation_vs_title_volatility_ratio": ratio,
                "rank_dynamics_data_available": True,
                "late_season_rounds_used": late_rounds_note,
            }
        )

        aud = sub_valid.copy()
        aud["round_from"] = aud["round_from"].astype("Int64")
        aud["round_to"] = aud["round_to"].astype("Int64")
        aud["absolute_rank_change"] = aud["abs_change"]
        aud = aud.rename(columns={"stage": "stage_from"})
        aud["stage_to"] = aud["stage_from"]
        audit_parts.append(
            aud[
                [
                    "season",
                    "round_from",
                    "round_to",
                    "team",
                    "prev_rank",
                    "new_rank",
                    "absolute_rank_change",
                    "stage_from",
                    "stage_to",
                ]
            ]
        )

    summary = pd.DataFrame(summary_rows)
    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    return summary, audit


def compute_combined_competitiveness(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add normalized components and combined score; return ranking table."""
    df = summary.copy()
    for col in (
        "title_race_closeness_score",
        "relegation_closeness_score",
        "league_balance_score",
    ):
        if col not in df.columns:
            df[col] = np.nan

    df["title_race_closeness_normalized"] = _minmax_normalize(
        df["title_race_closeness_score"], higher_is_better=True
    )
    df["relegation_closeness_normalized"] = _minmax_normalize(
        df["relegation_closeness_score"], higher_is_better=True
    )
    df["league_balance_normalized"] = _minmax_normalize(
        df["league_balance_score"], higher_is_better=True
    )

    norm_cols = [
        "title_race_closeness_normalized",
        "relegation_closeness_normalized",
        "league_balance_normalized",
    ]
    df["components_available"] = df[norm_cols].notna().sum(axis=1)
    df["combined_competitiveness_score"] = df[norm_cols].mean(axis=1, skipna=True)
    df.loc[df["components_available"] < 2, "combined_competitiveness_score"] = np.nan

    ranked = df.dropna(subset=["combined_competitiveness_score"]).copy()
    ranked["combined_competitiveness_rank"] = (
        ranked["combined_competitiveness_score"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    df = df.merge(
        ranked[["season", "combined_competitiveness_rank"]],
        on="season",
        how="left",
    )

    ranking_out = df[
        [
            "season",
            "title_race_closeness_normalized",
            "relegation_closeness_normalized",
            "league_balance_normalized",
            "combined_competitiveness_score",
            "combined_competitiveness_rank",
            "components_available",
        ]
    ].copy()
    return df, ranking_out


def _leader_by_round(df: pd.DataFrame) -> list[tuple[int, str]]:
    leaders = []
    for rnd in sorted(df["round"].unique()):
        tbl = df[df["round"] == rnd].sort_values("position")
        if tbl.empty:
            continue
        leaders.append((int(rnd), str(tbl.iloc[0]["team"])))
    return leaders


def _relegation_zone_teams(df: pd.DataFrame, rnd: int, n_rel: int = 2) -> set[str]:
    tbl = df[df["round"] == rnd].sort_values("position")
    if tbl.empty:
        return set()
    n = len(tbl)
    cutoff = max(n - n_rel + 1, 1)
    return set(tbl[tbl["position"] >= cutoff]["team"].astype(str))


def compute_playoff_impact(
    seasons: list[str],
    is_playoff_season_fn: Callable[[str], bool],
    load_cp_fn: Callable[[str], pd.DataFrame | None],
    load_rp_fn: Callable[[str], pd.DataFrame | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for season in seasons:
        row: dict[str, Any] = {
            "season": season,
            "championship_playoff_leadership_changes": np.nan,
            "championship_playoff_first_place_flipped": np.nan,
            "relegation_playoff_zone_changes": np.nan,
            "relegation_playoff_survival_flips": np.nan,
            "playoff_impact_balance": "no_playoff_or_insufficient_data",
            "relegation_vs_championship_playoff_impact_ratio": np.nan,
            "playoff_data_completeness_flag": "no_playoff",
        }
        audit: dict[str, Any] = {
            "season": season,
            "playoff_format": "no_playoff",
            "championship_start_leader": "",
            "championship_end_leader": "",
            "championship_playoff_leadership_changes": np.nan,
            "championship_playoff_first_place_flipped": np.nan,
            "relegation_start_zone_teams": "",
            "relegation_end_zone_teams": "",
            "relegation_playoff_zone_changes": np.nan,
            "relegation_playoff_survival_flips": np.nan,
            "playoff_impact_balance": "no_playoff_or_insufficient_data",
        }

        if not is_playoff_season_fn(season):
            rows.append(row)
            audit_rows.append(audit)
            continue

        cp = load_cp_fn(season)
        rp = load_rp_fn(season)
        has_cp = cp is not None and not cp.empty
        has_rp = rp is not None and not rp.empty

        if has_cp and has_rp:
            row["playoff_data_completeness_flag"] = "complete"
            audit["playoff_format"] = "championship_and_relegation"
        elif has_cp or has_rp:
            row["playoff_data_completeness_flag"] = "partial"
            audit["playoff_format"] = "partial"
        else:
            row["playoff_data_completeness_flag"] = "missing"
            audit["playoff_format"] = "missing"
            rows.append(row)
            audit_rows.append(audit)
            continue

        ch_impact = 0.0
        if has_cp:
            leaders = _leader_by_round(cp)
            if len(leaders) >= 2:
                ch_changes = sum(
                    1 for i in range(1, len(leaders)) if leaders[i][1] != leaders[i - 1][1]
                )
                row["championship_playoff_leadership_changes"] = ch_changes
                flipped = leaders[0][1] != leaders[-1][1]
                row["championship_playoff_first_place_flipped"] = bool(flipped)
                audit["championship_start_leader"] = leaders[0][1]
                audit["championship_end_leader"] = leaders[-1][1]
                audit["championship_playoff_leadership_changes"] = ch_changes
                audit["championship_playoff_first_place_flipped"] = bool(flipped)
                ch_impact = ch_changes + (1.0 if flipped else 0.0)

        rel_impact = 0.0
        if has_rp:
            rounds = sorted(rp["round"].unique())
            if len(rounds) >= 2:
                zone_changes = 0
                prev_zone: set[str] | None = None
                for rnd in rounds:
                    zone = _relegation_zone_teams(rp, rnd)
                    if prev_zone is not None:
                        zone_changes += len(prev_zone ^ zone)
                    prev_zone = zone

                start_zone = _relegation_zone_teams(rp, rounds[0])
                end_zone = _relegation_zone_teams(rp, rounds[-1])
                survived = len(start_zone - end_zone)
                dropped = len(end_zone - start_zone)
                survival_flips = survived + dropped

                row["relegation_playoff_zone_changes"] = zone_changes
                row["relegation_playoff_survival_flips"] = survival_flips
                audit["relegation_start_zone_teams"] = "; ".join(sorted(start_zone))
                audit["relegation_end_zone_teams"] = "; ".join(sorted(end_zone))
                audit["relegation_playoff_zone_changes"] = zone_changes
                audit["relegation_playoff_survival_flips"] = survival_flips
                rel_impact = zone_changes + survival_flips

        if ch_impact > 0 or rel_impact > 0:
            row["relegation_vs_championship_playoff_impact_ratio"] = round(
                rel_impact / max(1.0, ch_impact), 3
            )
            if rel_impact > ch_impact * 1.25:
                balance = "relegation_more_impactful"
            elif ch_impact > rel_impact * 1.25:
                balance = "championship_more_impactful"
            else:
                balance = "similar_impact"
            row["playoff_impact_balance"] = balance
            audit["playoff_impact_balance"] = balance

        rows.append(row)
        audit_rows.append(audit)

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def _budget_coverage_by_season(seasons: list[str]) -> dict[str, float | None]:
    if not BUDGET_PATH.exists():
        return {s: None for s in seasons}
    bud = pd.read_csv(BUDGET_PATH)
    out: dict[str, float | None] = {}
    for season in seasons:
        sub = bud[bud["season"] == season]
        if sub.empty:
            out[season] = None
            continue
        has = sub["budget_mid_million_nis"].notna() | sub["budget_min_million_nis"].notna()
        out[season] = round(100 * has.mean(), 1)
    return out


def _player_assists_available(season: str) -> bool | None:
    if season in ASSISTS_UNAVAILABLE_SEASONS:
        return False
    if assists_path_for_season is None or is_valid_stats_file is None:
        return None
    return bool(is_valid_stats_file(assists_path_for_season(season), "assists"))


def compute_data_quality(summary: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    budget_cov = _budget_coverage_by_season(seasons)
    rows: list[dict[str, Any]] = []

    for season in seasons:
        sub = summary[summary["season"] == season]
        r = sub.iloc[0].to_dict() if not sub.empty else {}

        att_cov = r.get("attendance_coverage_percentage")
        goals_ok = bool(r.get("player_stats_goals_available", False))
        assists_ok = _player_assists_available(season)
        playoff_flag = r.get("playoff_data_completeness_flag", "no_playoff")
        rank_ok = bool(r.get("rank_dynamics_data_available", False))
        bud_pct = budget_cov.get(season)

        score = 0.0
        notes: list[str] = []

        if pd.notna(att_cov):
            score += min(float(att_cov), 100) / 100 * 25
        else:
            notes.append("attendance coverage unknown")
        if goals_ok:
            score += 20
        else:
            notes.append("scorer data missing")
        if assists_ok is True:
            score += 15
        elif assists_ok is False:
            notes.append("assists unavailable on Transfermarkt")
        if playoff_flag == "complete":
            score += 20
        elif playoff_flag == "partial":
            score += 10
            notes.append("playoff data partial")
        elif playoff_flag == "missing" and str(r.get("has_playoff_data", "")):
            notes.append("playoff tracking missing")
        if rank_ok:
            score += 20
        else:
            notes.append("rank dynamics unavailable")

        if pd.notna(att_cov) and float(att_cov) < 60:
            notes.append("attendance below threshold")

        rows.append(
            {
                "season": season,
                "attendance_coverage_percentage": att_cov,
                "player_stats_goals_available": goals_ok,
                "player_stats_assists_available": assists_ok,
                "budget_data_coverage_pct": bud_pct,
                "playoff_data_completeness_flag": playoff_flag,
                "rank_dynamics_data_available": rank_ok,
                "combined_data_quality_score": round(score, 1),
                "data_quality_notes": "; ".join(notes),
            }
        )

    return pd.DataFrame(rows)


def save_advanced_charts(summary: pd.DataFrame) -> list[str]:
    ADV_CHARTS.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    completed = summary[summary.get("is_completed_season", True)].copy()
    seasons = sorted(completed["season"].tolist(), key=lambda s: tuple(map(int, s.split("/"))))

    def _bar(col: str, title: str, fname: str, ylabel: str) -> None:
        sub = completed.dropna(subset=[col])
        if sub.empty:
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        xs = sub["season"].tolist()
        ys = sub[col].tolist()
        ax.bar(xs, ys, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        path = ADV_CHARTS / fname
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sub[["season", col]].to_csv(ADV_CHARTS / fname.replace(".png", ".csv"), index=False, encoding="utf-8-sig")
        created.append(str(path))

    _bar("top_scorer_goals", "Top Scorer Goals by Season", "01_top_scorer_goals_by_season.png", "Goals")
    _bar(
        "share_of_league_goals_by_top_3_scorers",
        "Top 3 Scorers Share of League Goals (%)",
        "02_top3_scorers_goal_share_by_season.png",
        "Share (%)",
    )
    _bar(
        "offensive_concentration_gini",
        "Offensive Concentration Gini by Season",
        "03_offensive_concentration_gini_by_season.png",
        "Gini",
    )

    sub = completed.dropna(subset=["avg_rank_changes_per_round"])
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w / 2, sub["avg_rank_changes_per_round"], w, label="Avg rank changes / round")
        ax.bar(x + w / 2, sub["late_season_rank_volatility"], w, label="Late-season volatility")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["season"], rotation=45, ha="right")
        ax.set_title("Rank Volatility by Season")
        ax.legend()
        plt.tight_layout()
        p = ADV_CHARTS / "04_rank_volatility_by_season.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sub[
            ["season", "avg_rank_changes_per_round", "late_season_rank_volatility"]
        ].to_csv(ADV_CHARTS / "04_rank_volatility_by_season.csv", index=False, encoding="utf-8-sig")
        created.append(str(p))

    sub = completed.dropna(subset=["title_zone_rank_volatility", "relegation_zone_rank_volatility"])
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w / 2, sub["title_zone_rank_volatility"], w, label="Title zone (top 4)")
        ax.bar(x + w / 2, sub["relegation_zone_rank_volatility"], w, label="Relegation zone (bottom 4)")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["season"], rotation=45, ha="right")
        ax.set_title("Title vs Relegation Zone Rank Volatility (Late Season)")
        ax.legend()
        plt.tight_layout()
        p = ADV_CHARTS / "05_title_vs_relegation_volatility.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sub[
            ["season", "title_zone_rank_volatility", "relegation_zone_rank_volatility"]
        ].to_csv(ADV_CHARTS / "05_title_vs_relegation_volatility.csv", index=False, encoding="utf-8-sig")
        created.append(str(p))

    sub = completed.dropna(subset=["combined_competitiveness_score"]).sort_values(
        "combined_competitiveness_score", ascending=True
    )
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(sub["season"], sub["combined_competitiveness_score"], color="#59a14f")
        ax.set_title("Combined Competitiveness Score by Season")
        ax.set_xlabel("Score (0–100)")
        plt.tight_layout()
        p = ADV_CHARTS / "06_combined_competitiveness_ranking.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sub[["season", "combined_competitiveness_score", "combined_competitiveness_rank"]].to_csv(
            ADV_CHARTS / "06_combined_competitiveness_ranking.csv", index=False, encoding="utf-8-sig"
        )
        created.append(str(p))

    sub = completed[
        completed["playoff_impact_balance"].notna()
        & (completed["playoff_impact_balance"] != "no_playoff_or_insufficient_data")
    ]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(sub))
        w = 0.35
        ch = sub["championship_playoff_leadership_changes"].fillna(0)
        ch_flip = sub["championship_playoff_first_place_flipped"].map({True: 1, False: 0}).fillna(0)
        rel = sub["relegation_playoff_zone_changes"].fillna(0) + sub["relegation_playoff_survival_flips"].fillna(0)
        ax.bar(x - w / 2, ch + ch_flip, w, label="Championship playoff impact")
        ax.bar(x + w / 2, rel, w, label="Relegation playoff impact")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["season"], rotation=45, ha="right")
        ax.set_title("Playoff Impact Comparison by Season")
        ax.legend()
        plt.tight_layout()
        p = ADV_CHARTS / "07_playoff_impact_comparison.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sub[
            [
                "season",
                "championship_playoff_leadership_changes",
                "championship_playoff_first_place_flipped",
                "relegation_playoff_zone_changes",
                "relegation_playoff_survival_flips",
                "playoff_impact_balance",
            ]
        ].to_csv(ADV_CHARTS / "07_playoff_impact_comparison.csv", index=False, encoding="utf-8-sig")
        created.append(str(p))

    _bar(
        "combined_data_quality_score",
        "Combined Data Quality Score by Season",
        "08_data_quality_score_by_season.png",
        "Score (0–100)",
    )

    return created


def write_advanced_readme() -> str:
    text = """Advanced Competitiveness Features
===================================

Offensive concentration
- top_scorer_goals: goals by league top scorer (Transfermarkt scorers list)
- share_of_league_goals_by_top_3_scorers: top 3 scorers' goals / season total goals * 100
- offensive_concentration_gini: Gini over all players' goals (higher = more concentrated)
Sources: notebooks/data/raw/player_stats/top_scorers_*_ligat_haal_transfermarkt.csv

Rank dynamics
- Parsed from notebooks/data/processed/rank_dynamics/rank_changes_export.csv
- Late-season window = last 5 rank transitions in season timeline
- Title zone = prev_rank <= 4; relegation zone = bottom 4 in cohort

Combined competitiveness
- Mean of min-max normalized title_race_closeness, relegation_closeness, league_balance (0-100)
- Requires at least 2 of 3 components

Playoff impact
- Championship: leadership changes and first-place flip during championship playoff only
- Relegation: zone membership changes and survival flips during relegation playoff only
- Descriptive only; not a statistical test

Data quality
- combined_data_quality_score: attendance (25) + goals (20) + assists (15) + playoff (20) + rank dynamics (20)

Known limitations
- Assists unavailable for 2007/08 and 2008/09 on Transfermarkt
- Attendance incomplete for seasons with coverage < 60%
- Budget coverage is partial (demographics join)
- Playoff impact is descriptive unless tested statistically
- Friday/weekday attendance charts are unchanged by this module
"""
    path = ADV_CHARTS / "README_advanced_competitiveness_features.txt"
    ADV_CHARTS.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def apply_advanced_features(
    summary: pd.DataFrame,
    seasons: list[str],
    is_playoff_season_fn: Callable[[str], bool],
    load_cp_fn: Callable[[str], pd.DataFrame | None],
    load_rp_fn: Callable[[str], pd.DataFrame | None],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Merge all advanced feature groups into summary and write audit files.
    Returns (updated summary, dict of created file paths by group).
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    created: dict[str, list[str]] = {
        "offensive": [],
        "rank_dynamics": [],
        "combined": [],
        "playoff": [],
        "data_quality": [],
        "charts": [],
    }

    total_goals = {}
    if "total_goals" in summary.columns:
        for _, r in summary.iterrows():
            tg = r.get("total_goals")
            total_goals[r["season"]] = float(tg) if pd.notna(tg) and tg > 0 else None

    off_sum, off_audit = compute_offensive_concentration(seasons, total_goals)
    summary = summary.drop(columns=[c for c in off_sum.columns if c != "season" and c in summary.columns], errors="ignore")
    summary = summary.merge(off_sum, on="season", how="left")
    if not off_audit.empty:
        p = OUTPUT / "offensive_concentration_audit.csv"
        off_audit.to_csv(p, index=False, encoding="utf-8-sig")
        created["offensive"].append(str(p))

    rank_sum, rank_audit = compute_rank_dynamics(seasons)
    drop = [c for c in rank_sum.columns if c != "season" and c in summary.columns]
    summary = summary.drop(columns=drop, errors="ignore")
    summary = summary.merge(rank_sum, on="season", how="left")
    if not rank_audit.empty:
        p = OUTPUT / "rank_dynamics_audit.csv"
        rank_audit.to_csv(p, index=False, encoding="utf-8-sig")
        created["rank_dynamics"].append(str(p))

    summary, comb_rank = compute_combined_competitiveness(summary)
    p = OUTPUT / "combined_competitiveness_ranking.csv"
    comb_rank.to_csv(p, index=False, encoding="utf-8-sig")
    created["combined"].append(str(p))

    po_sum, po_audit = compute_playoff_impact(seasons, is_playoff_season_fn, load_cp_fn, load_rp_fn)
    drop = [c for c in po_sum.columns if c != "season" and c in summary.columns]
    summary = summary.drop(columns=drop, errors="ignore")
    summary = summary.merge(po_sum, on="season", how="left")
    if not po_audit.empty:
        p = OUTPUT / "playoff_impact_comparison_audit.csv"
        po_audit.to_csv(p, index=False, encoding="utf-8-sig")
        created["playoff"].append(str(p))

    dq = compute_data_quality(summary, seasons)
    drop = [c for c in dq.columns if c != "season" and c in summary.columns]
    summary = summary.drop(columns=drop, errors="ignore")
    summary = summary.merge(dq, on="season", how="left")
    p = OUTPUT / "data_quality_summary.csv"
    dq.to_csv(p, index=False, encoding="utf-8-sig")
    created["data_quality"].append(str(p))

    created["charts"] = save_advanced_charts(summary)
    created["charts"].append(write_advanced_readme())

    return summary, created
