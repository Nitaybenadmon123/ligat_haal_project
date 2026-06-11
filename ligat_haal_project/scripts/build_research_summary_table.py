"""
Build season-level research summary table for Ligat Ha'Al final project.

Season-level research summary including match-level attendance where coverage is sufficient.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "data"
NOTEBOOKS = ROOT / "notebooks"
OUTPUT = NOTEBOOKS / "outputs" / "research_summary"
CHARTS = OUTPUT / "charts"

COMPLETED_SEASON_END = "2025/26"
TRACKING_BACKUP_2024_25 = (
    DATA / "regular_season_tracking" / "regular_season_tracking_2024_25_bak_20260429_141341.csv"
)

# Reuse team normalization from enrichment script
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from enrich_transfermarkt_match_dates import TEAM_NAME_MAP  # type: ignore
except ImportError:
    TEAM_NAME_MAP = {}

try:
    from player_goal_contributions import apply_goal_contribution_to_summary  # type: ignore
except ImportError:
    apply_goal_contribution_to_summary = None  # type: ignore

try:
    from advanced_competitiveness_features import apply_advanced_features  # type: ignore
except ImportError:
    apply_advanced_features = None  # type: ignore

ADV_CHARTS = OUTPUT / "charts" / "advanced_competitiveness_features"

ATTENDANCE_PLACEHOLDER_COLUMNS = [
    "attendance_available",
    "attendance_valid_match_count",
    "attendance_total_match_count",
    "attendance_coverage_percentage",
    "average_match_attendance_valid",
    "median_match_attendance_valid",
    "attendance_avg_saturday",
    "attendance_avg_weekday",
    "attendance_saturday_weekday_diff",
    "attendance_saturday_weekday_ttest_pvalue",
    "attendance_saturday_match_count",
    "attendance_weekday_match_count",
    "attendance_saturday_vs_weekday_status",
    "attendance_by_weekday_anova_pvalue",
    "top_match_attendance_avg",
    "non_top_match_attendance_avg",
    "top_match_attendance_diff",
    "top_match_ttest_pvalue",
    "top_match_count",
    "non_top_match_count",
    "top_match_status",
    "attendance_notes",
]

ATTENDANCE_DIR = NOTEBOOKS / "attendance"
ATTENDANCE_MIN_COVERAGE_PCT = 60.0
ATTENDANCE_MIN_DAY_SAMPLES = 5
WEEKDAY_ORDER_EN = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

ATTENDANCE_STATUS = "placeholder_pending_attendance_validation"
ATTENDANCE_NOTES = (
    "Attendance analysis will be added after match-level attendance data is "
    "fully collected and validated."
)

SCORE_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_team(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)


def parse_score(score: Any) -> tuple[float | None, float | None]:
    if pd.isna(score):
        return None, None
    m = SCORE_RE.match(str(score).strip())
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def season_sort_key(season: str) -> tuple[int, int]:
    y1, y2 = season.split("/")
    return int(y1), int(y2)


def list_seasons_from_matches(matches: pd.DataFrame) -> list[str]:
    seasons = sorted(matches["season"].dropna().unique(), key=season_sort_key)
    return [s for s in seasons if season_sort_key(s) >= (2006, 7)]


def relegation_spots(num_teams: int) -> int:
    """Relegation spots for survival-line calculation (default 2; format data unavailable)."""
    return 2


def calculate_gini(points: np.ndarray) -> float:
    arr = np.sort(points.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return np.nan
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def mean_valid_score(components: list[float | None]) -> float | None:
    vals = [c for c in components if c is not None and not pd.isna(c)]
    return float(np.mean(vals) * 100) if vals else np.nan


def tracking_path(season: str) -> Path | None:
    slug = season.replace("/", "_")
    primary = DATA / "regular_season_tracking" / f"regular_season_tracking_{slug}.csv"
    if primary.exists():
        return primary
    if season == "2024/25" and TRACKING_BACKUP_2024_25.exists():
        return TRACKING_BACKUP_2024_25
    return None


def load_tracking(season: str) -> tuple[pd.DataFrame | None, str]:
    path = tracking_path(season)
    if path is None:
        return None, "missing_tracking"
    df = pd.read_csv(path)
    df["team"] = df["team"].map(normalize_team)
    flag = "ok"
    if season == "2024/25" and "bak" in path.name:
        flag = "tracking_backup_2024_25"
    return df, flag


def load_final_table(season: str) -> pd.DataFrame | None:
    slug = season.replace("/", "_")
    path = DATA / "interim" / "scraped_standings" / "regular_final_tables" / f"regular_final_table_{slug}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "team" in df.columns:
        df["team"] = df["team"].map(normalize_team)
    return df


def load_processed_csv(rel_path: str) -> pd.DataFrame | None:
    path = DATA / "processed" / rel_path
    if not path.exists():
        return None
    return pd.read_csv(path)


def is_playoff_season(season: str) -> bool:
    """True for seasons with a championship playoff structure (2009/10+)."""
    ss = load_processed_csv("title_race_analysis/season_summary_statistics.csv")
    if ss is not None:
        row = ss[ss["season"] == season]
        if not row.empty:
            fmt = row.iloc[0].get("playoff_format")
            return pd.notna(fmt) and fmt != "No Playoff"
    return season_sort_key(season) >= (2009, 10)


def load_playoff_final_table(season: str) -> pd.DataFrame | None:
    """
    Championship playoff final standings (end-of-playoff table).
    Primary source: interim/scraped_standings/playoff_final_tables/
    Fallback: last round of championship_playoff_tracking/
    """
    slug = season.replace("/", "_")
    path = (
        DATA
        / "interim"
        / "scraped_standings"
        / "playoff_final_tables"
        / f"playoff_final_table_{slug}.csv"
    )
    if path.exists():
        df = pd.read_csv(path)
        if "team" in df.columns:
            df["team"] = df["team"].map(normalize_team)
        if "rank" in df.columns:
            return df.sort_values("rank").reset_index(drop=True)
        return df.sort_values("points", ascending=False).reset_index(drop=True)

    cp_path = DATA / "championship_playoff_tracking" / f"championship_playoff_tracking_{slug}.csv"
    if cp_path.exists():
        cp = pd.read_csv(cp_path)
        cp["team"] = cp["team"].map(normalize_team)
        last_round = cp["round"].max()
        last = cp[cp["round"] == last_round].sort_values("position").copy()
        last["rank"] = last["position"]
        return last[["rank", "team", "points"]].reset_index(drop=True)

    return None


def final_standings_from_table(
    table: pd.DataFrame,
) -> tuple[str, Any, float, Any, float, float]:
    """Return champion, runner_up, points, and gap from a sorted final standings table."""
    if "rank" in table.columns:
        ft = table.sort_values("rank")
    else:
        ft = table.sort_values("points", ascending=False)
    champion = ft.iloc[0]["team"]
    runner_up = ft.iloc[1]["team"] if len(ft) > 1 else np.nan
    champ_pts = float(ft.iloc[0]["points"])
    runner_pts = float(ft.iloc[1]["points"]) if len(ft) > 1 else np.nan
    final_gap = champ_pts - runner_pts if len(ft) > 1 else np.nan
    return champion, runner_up, champ_pts, runner_pts, final_gap


# ---------------------------------------------------------------------------
# Standings / title race from tracking
# ---------------------------------------------------------------------------
def build_round_tables(tracking: pd.DataFrame) -> dict[int, pd.DataFrame]:
    reg = tracking[tracking["stage"] == "regular"].copy()
    tables: dict[int, pd.DataFrame] = {}
    for rnd, grp in reg.groupby("round"):
        tbl = grp.sort_values(["position", "goal_diff", "points"], ascending=[True, False, False]).copy()
        tables[int(rnd)] = tbl.reset_index(drop=True)
    return tables


def leader_per_round(tables: dict[int, pd.DataFrame]) -> dict[int, str]:
    leaders = {}
    for rnd, tbl in tables.items():
        if tbl.empty:
            continue
        leaders[rnd] = tbl.iloc[0]["team"]
    return leaders


def matches_remaining_at_round(round_num: int, num_teams: int) -> int:
    """Legacy helper used by relegation decision approximation (not title decision)."""
    total_per_team = num_teams - 1
    return max(total_per_team - round_num, 0)


def load_championship_playoff_tracking(season: str) -> pd.DataFrame | None:
    slug = season.replace("/", "_")
    path = DATA / "championship_playoff_tracking" / f"championship_playoff_tracking_{slug}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["team"] = df["team"].map(normalize_team)
    return df


def load_relegation_playoff_tracking(season: str) -> pd.DataFrame | None:
    slug = season.replace("/", "_")
    path = DATA / "relegation_playoff_tracking" / f"relegation_playoff_tracking_{slug}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["team"] = df["team"].map(normalize_team)
    return df


def build_stage_round_tables(df: pd.DataFrame, stage_value: str) -> dict[int, pd.DataFrame]:
    """Build per-round sorted tables for a given stage column value."""
    sub = df[df["stage"] == stage_value].copy()
    tables: dict[int, pd.DataFrame] = {}
    for rnd, grp in sub.groupby("round"):
        tbl = grp.sort_values(["position", "goal_diff", "points"], ascending=[True, False, False]).copy()
        tables[int(rnd)] = tbl.reset_index(drop=True)
    return tables


def build_title_race_timeline(
    season: str,
    tracking: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], int, int, str]:
    """
    Combined title-race standings by global round.

    Returns:
        combined_tables, total_rounds_in_title_race_season, regular_rounds, title_decision_method
    """
    regular_tables = build_round_tables(tracking)
    regular_rounds = max(regular_tables) if regular_tables else 0

    if not is_playoff_season(season):
        if not regular_tables:
            return {}, 0, 0, "unavailable"
        return regular_tables, regular_rounds, regular_rounds, "full_season_no_playoff"

    cp_tracking = load_championship_playoff_tracking(season)
    if cp_tracking is None or cp_tracking.empty:
        if not regular_tables:
            return {}, 0, 0, "unavailable"
        return regular_tables, regular_rounds, regular_rounds, "regular_only_fallback"

    playoff_tables = build_stage_round_tables(cp_tracking, "championship_playoff")
    combined = dict(regular_tables)
    for rnd, tbl in playoff_tables.items():
        combined[int(rnd)] = tbl

    total_rounds = max(combined) if combined else 0
    return combined, total_rounds, regular_rounds, "full_season_with_championship_playoff"


def compute_title_decision_with_audit(
    season: str,
    combined_tables: dict[int, pd.DataFrame],
    total_rounds: int,
    regular_rounds: int,
    title_decision_method: str,
) -> tuple[int | None, float | None, list[dict[str, Any]]]:
    """
    Title is decided after round R when either:
        1. gap_1st_2nd > (total_rounds - R) * 3  (2nd place cannot catch up on points), or
        2. R is the final round (remaining_rounds == 0) — season over; ties go to tiebreakers.

    Before the final round, use strictly greater than (gap == points_available means a tie is
    still possible on points).
    """
    audit_rows: list[dict[str, Any]] = []
    title_decision_round_val: int | None = None

    if not combined_tables or total_rounds <= 0:
        return None, np.nan, audit_rows

    for global_round in sorted(combined_tables):
        if global_round > total_rounds:
            continue
        tbl = combined_tables[global_round]
        if tbl.empty or len(tbl) < 2:
            continue

        leader = tbl.iloc[0]["team"]
        runner_up = tbl.iloc[1]["team"]
        leader_pts = float(tbl.iloc[0]["points"])
        runner_up_pts = float(tbl.iloc[1]["points"])
        gap = leader_pts - runner_up_pts
        remaining_rounds = total_rounds - global_round
        points_available = remaining_rounds * 3
        is_decided = bool(gap > points_available or remaining_rounds == 0)

        if global_round <= regular_rounds:
            stage = "regular"
            local_round = global_round
        else:
            stage = "championship_playoff"
            local_round = global_round - regular_rounds

        audit_rows.append({
            "season": season,
            "stage": stage,
            "local_round": local_round,
            "global_round": global_round,
            "leader": leader,
            "leader_points": leader_pts,
            "runner_up": runner_up,
            "runner_up_points": runner_up_pts,
            "gap_1st_2nd": gap,
            "remaining_rounds": remaining_rounds,
            "points_available": points_available,
            "is_title_decided_after_round": is_decided,
            "title_decision_method": title_decision_method,
        })

        if is_decided and title_decision_round_val is None:
            title_decision_round_val = global_round

    rounds_before = (
        float(total_rounds - title_decision_round_val)
        if title_decision_round_val is not None
        else np.nan
    )
    return title_decision_round_val, rounds_before, audit_rows


# Max points behind leader to still count as "in the title race" at late-season checkpoints.
TITLE_RACE_MAX_GAP_POINTS = 7

# Contenders if abs(points - survival_line) <= band (symmetric).
# Survival line = first relegation spot (e.g. 11th in 12-team, 13th in 14-team, 15th in 16-team).
RELEGATION_SURVIVAL_BAND_PTS = 6
N_REL = 2  # relegated clubs per season (notebook default)
LEGACY_SINGLE_TABLE_SEASONS = frozenset({"2006/07", "2007/08", "2008/09"})


def teams_in_title_race_at_round(
    tables: dict[int, pd.DataFrame],
    rnd: int,
    max_gap_points: int = TITLE_RACE_MAX_GAP_POINTS,
) -> int | None:
    """
    Count teams within max_gap_points of the leader at round rnd.

    A team is in the title race if: leader_points - team_points <= max_gap_points
    (default 7 points at checkpoints 5/6/8 rounds before season end).
    """
    if rnd not in tables:
        return None
    tbl = tables[rnd]
    leader_pts = float(tbl.iloc[0]["points"])
    count = 0
    for _, row in tbl.iterrows():
        gap_to_leader = leader_pts - float(row["points"])
        if gap_to_leader <= max_gap_points:
            count += 1
    return count


def sort_standings_points_desc(tbl: pd.DataFrame) -> pd.DataFrame:
    """Sort standings like notebook Metric 1 (points descending)."""
    st = tbl.drop_duplicates(subset=["team"]).copy()
    return st.sort_values(
        ["points", "goal_diff", "team"], ascending=[False, False, True]
    ).reset_index(drop=True)


def relegation_zone_teams_snapshot(tbl: pd.DataFrame, n_rel: int = N_REL) -> set[str]:
    """Notebook Metric 4: clubs in zone when position >= N - n_rel + 1."""
    if tbl.empty:
        return set()
    st = tbl.drop_duplicates(subset=["team"]).copy()
    pos_num = pd.to_numeric(st["position"], errors="coerce")
    n_tab = len(st)
    if n_tab <= n_rel:
        return set()
    cut = n_tab - n_rel + 1
    return set(st.loc[pos_num >= cut, "team"].astype(str))


def survival_line_points_from_table(
    tbl: pd.DataFrame,
    n_rel: int = N_REL,
) -> float | None:
    """
    Points of the team at the first relegation spot in a points-sorted table.

    Examples (n_rel=2): 11th of 12, 13th of 14, 15th of 16, 7th of 8 in relegation playoff.
    """
    if tbl is None or tbl.empty:
        return None
    st = sort_standings_points_desc(tbl)
    n_table = len(st)
    if n_table <= n_rel:
        return None
    survival_idx = n_table - n_rel  # first relegation spot (0-based)
    return float(st.iloc[survival_idx]["points"])


def relegation_contenders_at_snapshot(
    tbl: pd.DataFrame,
    n_rel: int = N_REL,
    band_pts: int = RELEGATION_SURVIVAL_BAND_PTS,
) -> tuple[int | None, float | None, pd.DataFrame]:
    """
    Relegation contenders at one standings snapshot.

    - Survival line: first relegation spot at index (N - n_rel) in points-sorted table.
    - Contender if abs(points - survival) <= band_pts AND rank_snap >= N - 6.
    """
    if tbl is None or tbl.empty:
        return None, None, pd.DataFrame()

    st = sort_standings_points_desc(tbl)
    n_table = len(st)
    if n_table <= n_rel:
        return 0, None, st

    survival_pts = survival_line_points_from_table(tbl, n_rel)
    if survival_pts is None:
        return 0, None, st

    st = st.copy()
    st["rank_snap"] = np.arange(1, n_table + 1)
    st["gap_to_survival"] = (st["points"] - survival_pts).abs()
    rank_floor = n_table - 6
    contenders = st[
        (st["gap_to_survival"] <= band_pts) & (st["rank_snap"] >= rank_floor)
    ].copy()
    return len(contenders), survival_pts, contenders


def compute_title_race_metrics(season: str, tracking: pd.DataFrame, final_table: pd.DataFrame | None) -> dict[str, Any]:
    regular_tables = build_round_tables(tracking)
    regular_rounds = max(regular_tables) if regular_tables else np.nan
    regular_leaders = leader_per_round(regular_tables)

    combined_tables, total_title_rounds, reg_rounds_int, title_decision_method = build_title_race_timeline(
        season, tracking
    )
    combined_leaders = leader_per_round(combined_tables)
    # Leadership volatility/stability: full title timeline when championship playoff tracking exists
    leaders = (
        combined_leaders
        if title_decision_method == "full_season_with_championship_playoff"
        else regular_leaders
    )

    notes: list[str] = []
    gap_source = "regular_final_table"

    # Playoff seasons: champion, runner-up, and final_gap from championship playoff final table
    if is_playoff_season(season):
        playoff_final = load_playoff_final_table(season)
        if playoff_final is not None and "points" in playoff_final.columns:
            champion, runner_up, champ_pts, runner_pts, final_gap = final_standings_from_table(
                playoff_final
            )
            gap_source = "championship_playoff_final_table"
            notes.append(
                "champion, runner_up, final_gap_1st_2nd: from championship playoff final table"
            )
        elif final_table is not None and "points" in final_table.columns:
            champion, runner_up, champ_pts, runner_pts, final_gap = final_standings_from_table(
                final_table
            )
            notes.append(
                "Playoff season but no playoff final table; using regular-season final standings"
            )
        else:
            last_tbl = regular_tables.get(int(regular_rounds)) if regular_tables else None
            if last_tbl is None or last_tbl.empty:
                return {"notes": "No final standings available for title race."}
            champion = last_tbl.iloc[0]["team"]
            runner_up = last_tbl.iloc[1]["team"] if len(last_tbl) > 1 else np.nan
            champ_pts = float(last_tbl.iloc[0]["points"])
            runner_pts = float(last_tbl.iloc[1]["points"]) if len(last_tbl) > 1 else np.nan
            final_gap = champ_pts - runner_pts
            notes.append("Playoff season; fallback to last regular-season round standings")
    elif final_table is not None and "points" in final_table.columns:
        champion, runner_up, champ_pts, runner_pts, final_gap = final_standings_from_table(final_table)
    else:
        last_tbl = regular_tables.get(int(regular_rounds)) if regular_tables else None
        if last_tbl is None or last_tbl.empty:
            return {"notes": "No final standings available for title race."}
        champion = last_tbl.iloc[0]["team"]
        runner_up = last_tbl.iloc[1]["team"] if len(last_tbl) > 1 else np.nan
        champ_pts = float(last_tbl.iloc[0]["points"])
        runner_pts = float(last_tbl.iloc[1]["points"]) if len(last_tbl) > 1 else np.nan
        final_gap = champ_pts - runner_pts

    # Title decision on full title-race timeline (regular + championship playoff when available)
    tdr, rounds_before, title_audit_rows = compute_title_decision_with_audit(
        season,
        combined_tables,
        total_title_rounds,
        reg_rounds_int,
        title_decision_method,
    )
    if title_decision_method == "regular_only_fallback":
        notes.append(
            "title_decision_round: regular-season only fallback; may be inaccurate for playoff season"
        )
    elif title_decision_method == "full_season_with_championship_playoff":
        notes.append(
            "title_decision_round: computed on regular season + championship playoff timeline"
        )
        notes.append(
            "lead_changes_after_round_10 and champion leadership %: include championship playoff rounds"
        )

    # Teams in race near end (use global rounds on combined timeline)
    end_round = total_title_rounds if total_title_rounds else int(regular_rounds)
    t5 = (
        teams_in_title_race_at_round(combined_tables, end_round - 5)
        if end_round >= 5 and total_title_rounds
        else np.nan
    )
    t6 = (
        teams_in_title_race_at_round(combined_tables, end_round - 6)
        if end_round >= 6 and total_title_rounds
        else np.nan
    )
    t8 = (
        teams_in_title_race_at_round(combined_tables, end_round - 8)
        if end_round >= 8 and total_title_rounds
        else np.nan
    )
    last5_rounds = [end_round - i for i in range(5, 0, -1) if end_round - i >= 1]
    race_counts = [
        teams_in_title_race_at_round(combined_tables, r)
        for r in last5_rounds
    ]
    race_counts = [c for c in race_counts if c is not None]
    avg_race_last5 = float(np.mean(race_counts)) if race_counts else np.nan

    # Lead changes after round 10
    sorted_rounds = sorted(leaders)
    lead_changes_after_10 = 0
    for i in range(1, len(sorted_rounds)):
        r_prev, r_curr = sorted_rounds[i - 1], sorted_rounds[i]
        if r_curr <= 10:
            continue
        if leaders[r_prev] != leaders[r_curr]:
            lead_changes_after_10 += 1

    # Champion stability (leaders timeline above; first round always regular season round 1)
    champ_rounds_as_leader = sum(1 for l in leaders.values() if l == champion)
    first_round_leader = regular_leaders.get(1)
    first_round_champion_was_1st = first_round_leader == champion if first_round_leader else np.nan
    champ_pct = (champ_rounds_as_leader / len(leaders) * 100) if leaders else np.nan

    # Longest consecutive streak by champion
    longest_streak = 0
    current = 0
    for rnd in sorted_rounds:
        if leaders[rnd] == champion:
            current += 1
            longest_streak = max(longest_streak, current)
        else:
            current = 0

    # Closeness score (decision timing relative to full title-race season length)
    gap_score = 1 - min(final_gap / 20, 1) if pd.notna(final_gap) else None
    decision_score = (
        tdr / total_title_rounds if tdr and total_title_rounds else None
    )
    race_score = min(t5 / 6, 1) if pd.notna(t5) else None
    lead_score = min(lead_changes_after_10 / 10, 1)
    closeness = mean_valid_score([gap_score, decision_score, race_score, lead_score])

    return {
        "champion": champion,
        "runner_up": runner_up,
        "champion_points": champ_pts,
        "runner_up_points": runner_pts,
        "final_gap_1st_2nd": final_gap,
        "title_decision_round": tdr,
        "rounds_before_end_when_title_decided": rounds_before,
        "title_decision_method": title_decision_method,
        "total_rounds_in_title_race": total_title_rounds,
        "title_audit_rows": title_audit_rows,
        "teams_in_title_race_5_rounds_before_end": t5,
        "teams_in_title_race_6_rounds_before_end": t6,
        "teams_in_title_race_8_rounds_before_end": t8,
        "average_title_race_teams_last_5_rounds": avg_race_last5,
        "lead_changes_after_round_10": lead_changes_after_10,
        "first_round_champion_was_1st": first_round_champion_was_1st,
        "number_of_rounds_champion_was_1st": champ_rounds_as_leader,
        "champion_rounds_as_leader_percentage": champ_pct,
        "longest_consecutive_lead_streak_by_champion": longest_streak,
        "title_race_closeness_score": closeness,
        "regular_rounds": regular_rounds,
        "final_gap_source": gap_source,
        "title_notes": "; ".join(notes) if notes else "",
    }


# ---------------------------------------------------------------------------
# Relegation race timeline (regular + relegation playoff when available)
# ---------------------------------------------------------------------------
def _stage_for_global_round(global_round: int, regular_rounds: int) -> tuple[str, int]:
    if global_round <= regular_rounds:
        return "regular", global_round
    return "relegation_playoff", global_round - regular_rounds


def build_relegation_race_timeline(
    season: str,
    tracking: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], int, int, str]:
    """
    Combined relegation-race standings by global round.

    Returns:
        regular_tables, combined_tables, total_rounds, regular_rounds, relegation_race_method
    """
    regular_tables = build_round_tables(tracking)
    regular_rounds = max(regular_tables) if regular_tables else 0

    if not is_playoff_season(season):
        if not regular_tables:
            return {}, {}, 0, 0, "unavailable"
        return regular_tables, regular_tables, regular_rounds, regular_rounds, "full_season_no_playoff"

    rp_tracking = load_relegation_playoff_tracking(season)
    if rp_tracking is None or rp_tracking.empty:
        if not regular_tables:
            return {}, {}, 0, 0, "unavailable"
        return regular_tables, regular_tables, regular_rounds, regular_rounds, "regular_only_fallback"

    playoff_tables = build_stage_round_tables(rp_tracking, "relegation_playoff")
    combined = dict(regular_tables)
    for rnd, tbl in playoff_tables.items():
        combined[int(rnd)] = tbl

    total_rounds = max(combined) if combined else 0
    return regular_tables, combined, total_rounds, regular_rounds, "full_season_with_relegation_playoff"


def teams_in_relegation_race_at_round(
    tables: dict[int, pd.DataFrame],
    rnd: int,
    band_pts: int = RELEGATION_SURVIVAL_BAND_PTS,
) -> int | None:
    """Count relegation contenders at round rnd (notebook Metric 1 definition)."""
    if rnd not in tables:
        return None
    count, _, _ = relegation_contenders_at_snapshot(tables[rnd], band_pts=band_pts)
    return count


def compute_relegation_zone_volatility(
    combined_tables: dict[int, pd.DataFrame],
    regular_rounds: int,
    n_rel: int = N_REL,
) -> tuple[int, int, int]:
    """Notebook Metric 4: half symmetric-diff zone changes on full relegation timeline."""
    rnd_list = sorted(combined_tables)
    if len(rnd_list) < 2:
        return 0, 0, 0

    zone_sets = [
        relegation_zone_teams_snapshot(combined_tables[r], n_rel) for r in rnd_list
    ]
    total_changes = 0
    after_10_changes = 0
    last5_changes = 0
    end_round = rnd_list[-1]

    for i in range(len(zone_sets) - 1):
        r_curr = rnd_list[i + 1]
        diff = len(zone_sets[i].symmetric_difference(zone_sets[i + 1])) // 2
        total_changes += diff
        if r_curr > 10:
            after_10_changes += diff
        if r_curr > end_round - 5:
            last5_changes += diff

    return total_changes, after_10_changes, last5_changes


def build_relegation_race_audit(
    season: str,
    combined_tables: dict[int, pd.DataFrame],
    total_rounds: int,
    regular_rounds: int,
    relegation_race_method: str,
    band_pts: int = RELEGATION_SURVIVAL_BAND_PTS,
) -> list[dict[str, Any]]:
    """One audit row per team per round on the full relegation timeline."""
    audit_rows: list[dict[str, Any]] = []
    if not combined_tables or total_rounds <= 0:
        return audit_rows

    for global_round in sorted(combined_tables):
        if global_round > total_rounds:
            continue
        tbl = combined_tables[global_round]
        if tbl.empty:
            continue

        stage, local_round = _stage_for_global_round(global_round, regular_rounds)
        _, line_pts, _ = relegation_contenders_at_snapshot(tbl, band_pts=band_pts)
        if line_pts is None:
            continue

        rank_floor = len(sort_standings_points_desc(tbl)) - 6
        st_pts = sort_standings_points_desc(tbl)
        st_pts = st_pts.copy()
        st_pts["rank_snap"] = np.arange(1, len(st_pts) + 1)
        st_pts["gap_to_survival"] = (st_pts["points"] - line_pts).abs()
        pos_map = dict(zip(tbl["team"].astype(str), pd.to_numeric(tbl["position"], errors="coerce")))

        for _, row in st_pts.iterrows():
            pts = float(row["points"])
            gap = float(row["gap_to_survival"])
            rank_snap = int(row["rank_snap"])
            in_band = gap <= band_pts
            in_bottom_block = rank_snap >= rank_floor
            tm_pos = pos_map.get(str(row["team"]), np.nan)
            audit_rows.append({
                "season": season,
                "stage": stage,
                "local_round": local_round,
                "global_round": global_round,
                "team": row["team"],
                "rank_snap": rank_snap,
                "position": int(tm_pos) if pd.notna(tm_pos) else rank_snap,
                "points": pts,
                "survival_line_points": line_pts,
                "gap_to_survival": gap,
                "rank_floor": rank_floor,
                "is_in_relegation_race": bool(in_band and in_bottom_block),
                "relegation_race_method": relegation_race_method,
            })

    return audit_rows


# ---------------------------------------------------------------------------
# Relegation metrics
# ---------------------------------------------------------------------------
def compute_relegation_metrics(season: str, tracking: pd.DataFrame) -> dict[str, Any]:
    regular_tables, combined_tables, total_rounds, reg_rounds_int, race_method = (
        build_relegation_race_timeline(season, tracking)
    )
    regular_rounds = reg_rounds_int if reg_rounds_int else np.nan

    # Volatility: notebook Metric 4 (regular + relegation playoff, half symmetric diff)
    total_changes, after_10_changes, last5_changes = compute_relegation_zone_volatility(
        combined_tables, reg_rounds_int
    )

    # Validate against processed metric4 when available
    metric4 = load_processed_csv("relegation_competitiveness/metric4_relegation_zone_volatility.csv")
    source_total = np.nan
    if metric4 is not None:
        row = metric4[metric4["season"] == season]
        if not row.empty:
            source_total = int(row.iloc[0]["total_relegation_zone_changes"])
            if source_total != total_changes:
                warnings.warn(
                    f"{season}: recomputed relegation_zone_changes_total={total_changes} "
                    f"differs from metric4={source_total}; using recomputed value."
                )

    def regular_contenders_at(round_num: int) -> int | None:
        return teams_in_relegation_race_at_round(regular_tables, round_num)

    def full_contenders_at(round_num: int) -> int | None:
        return teams_in_relegation_race_at_round(combined_tables, round_num)

    reg_c5 = (
        regular_contenders_at(reg_rounds_int - 5)
        if reg_rounds_int >= 5
        else np.nan
    )
    reg_c6 = (
        regular_contenders_at(reg_rounds_int - 6)
        if reg_rounds_int >= 6
        else np.nan
    )
    reg_c8 = (
        regular_contenders_at(reg_rounds_int - 8)
        if reg_rounds_int >= 8
        else np.nan
    )

    end_round = total_rounds if total_rounds else reg_rounds_int
    full_c5 = full_contenders_at(end_round - 5) if end_round >= 5 and total_rounds else np.nan
    full_c6 = full_contenders_at(end_round - 6) if end_round >= 6 and total_rounds else np.nan
    full_c8 = full_contenders_at(end_round - 8) if end_round >= 8 and total_rounds else np.nan

    # No-playoff seasons: full-season metrics equal regular-season metrics
    if race_method == "full_season_no_playoff":
        full_c5, full_c6, full_c8 = reg_c5, reg_c6, reg_c8

    relegation_audit_rows = build_relegation_race_audit(
        season,
        combined_tables,
        total_rounds,
        reg_rounds_int,
        race_method,
    )

    # Relegation decision round: prefer notebook Metric 2 output when available
    rel_decision = None
    decision_method = "unavailable"
    metric2 = load_processed_csv("relegation_competitiveness/metric2_relegation_decision_timing.csv")
    if metric2 is not None:
        sub = metric2[metric2["season"] == season]
        if not sub.empty:
            elim = sub["elimination_round"].dropna()
            if not elim.empty:
                rel_decision = int(elim.max())
                decision_method = "metric2_notebook"

    # Closeness uses full-season last-5-rounds checkpoint when available
    late_score = min(full_c5 / 6, 1) if pd.notna(full_c5) else None
    vol_score = min(total_changes / 15, 1)
    dec_score = rel_decision / regular_rounds if rel_decision and pd.notna(regular_rounds) else None
    rel_closeness = mean_valid_score([late_score, vol_score, dec_score])

    return {
        "teams_in_relegation_race_last_5_regular_rounds": reg_c5,
        "teams_in_relegation_race_last_6_regular_rounds": reg_c6,
        "teams_in_relegation_race_last_8_regular_rounds": reg_c8,
        "teams_in_relegation_race_last_5_rounds": full_c5,
        "teams_in_relegation_race_last_6_rounds": full_c6,
        "teams_in_relegation_race_last_8_rounds": full_c8,
        "relegation_race_method": race_method,
        "total_rounds_in_relegation_race": total_rounds,
        "relegation_audit_rows": relegation_audit_rows,
        "relegation_zone_changes_total": total_changes,
        "relegation_zone_changes_after_round_10": after_10_changes,
        "relegation_zone_changes_last_5_rounds": last5_changes,
        "relegation_decision_round": rel_decision,
        "relegation_decision_method": decision_method,
        "relegation_closeness_score": rel_closeness,
        "relegation_source_metric4_total": source_total,
    }


# ---------------------------------------------------------------------------
# Match-level metrics
# ---------------------------------------------------------------------------
def compute_match_metrics(season: str, matches: pd.DataFrame, regular_rounds: int | float) -> dict[str, Any]:
    sub = matches[matches["season"] == season].copy()
    sub[["home_goals", "away_goals"]] = sub["score"].apply(
        lambda x: pd.Series(parse_score(x))
    )
    valid = sub.dropna(subset=["home_goals", "away_goals"])
    invalid_count = len(sub) - len(valid)

    if valid.empty:
        return {
            "total_matches": len(sub),
            "valid_matches_for_analysis": 0,
            "match_notes": "No valid scores",
        }

    valid = valid.copy()
    valid["home_goals"] = valid["home_goals"].astype(float)
    valid["away_goals"] = valid["away_goals"].astype(float)
    valid["total_goals"] = valid["home_goals"] + valid["away_goals"]
    valid["home_win"] = (valid["home_goals"] > valid["away_goals"]).astype(int)
    valid["away_win"] = (valid["home_goals"] < valid["away_goals"]).astype(int)
    valid["draw"] = (valid["home_goals"] == valid["away_goals"]).astype(int)

    n = len(valid)
    notes = []
    if invalid_count:
        notes.append(f"{invalid_count} matches excluded (invalid score)")

    reg = valid
    if pd.notna(regular_rounds):
        reg = valid[valid["round"] <= regular_rounds]

    po = pd.DataFrame()
    if pd.notna(regular_rounds):
        po = valid[valid["round"] > regular_rounds]

    def agg_stage(df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {}
        m = len(df)
        return {
            "avg_goals": df["total_goals"].mean(),
            "home_win_pct": df["home_win"].mean() * 100,
        }

    # Supplement playoff from scraped playoff data if main TM file has no playoff rounds
    if po.empty and is_playoff_season(season):
        po_all = load_playoff_matches()
        po_season = po_all[po_all["season"] == season]
        if not po_season.empty:
            po = po_season
            notes.append("playoff matches supplemented from scraped playoff data")

    reg_agg = agg_stage(reg)
    po_agg = agg_stage(po)

    po_status = "computed"
    if not po_agg:
        po_status = "insufficient_stage_data"

    # Season-level match stats: regular + both playoffs when the season has playoffs
    if is_playoff_season(season) and not po.empty:
        full_pool = pd.concat([reg, po], ignore_index=True)
        notes.append("season match stats include regular + playoff games")
    else:
        full_pool = valid

    return {
        "total_matches": len(sub),
        "valid_matches_for_analysis": len(full_pool),
        "home_win_percentage": full_pool["home_win"].mean() * 100,
        "away_win_percentage": full_pool["away_win"].mean() * 100,
        "draw_percentage": full_pool["draw"].mean() * 100,
        "average_home_goals": full_pool["home_goals"].mean(),
        "average_away_goals": full_pool["away_goals"].mean(),
        "goal_difference_home_minus_away": (full_pool["home_goals"] - full_pool["away_goals"]).mean(),
        "total_goals": full_pool["total_goals"].sum(),
        "average_goals_per_match": full_pool["total_goals"].mean(),
        "matches_with_0_0": int((full_pool["total_goals"] == 0).sum()),
        "percentage_matches_over_2_5_goals": (full_pool["total_goals"] > 2.5).mean() * 100,
        "regular_average_goals_per_match": reg_agg.get("avg_goals", np.nan),
        "playoff_average_goals_per_match": po_agg.get("avg_goals", np.nan),
        "regular_home_win_percentage": reg_agg.get("home_win_pct", np.nan),
        "playoff_home_win_percentage": po_agg.get("home_win_pct", np.nan),
        "playoff_vs_regular_competitiveness_status": po_status,
        "match_notes": "; ".join(notes),
    }


def _add_match_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["home_goals", "away_goals"])
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["home_win"] = (out["home_goals"] > out["away_goals"]).astype(int)
    out["away_win"] = (out["home_goals"] < out["away_goals"]).astype(int)
    out["draw"] = (out["home_goals"] == out["away_goals"]).astype(int)
    return out


def load_playoff_matches() -> pd.DataFrame:
    """Championship + relegation playoff matches from BetExplorer and Transfermarkt round files."""
    frames: list[pd.DataFrame] = []
    playoff_dir = DATA / "playoffs" / "scraped_betexplorer"

    for fname in ["all_championship_betexplorer.csv", "all_relegation_betexplorer.csv"]:
        path = playoff_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(_add_match_outcome_columns(df[["season", "home_goals", "away_goals"]]))

    for path in sorted(playoff_dir.glob("matches_*_ligat_haal_*_round_transfermarkt.csv")):
        df = pd.read_csv(path)
        season_col = "Season" if "Season" in df.columns else "season"
        home_col = "HomeGoals" if "HomeGoals" in df.columns else "home_goals"
        away_col = "AwayGoals" if "AwayGoals" in df.columns else "away_goals"
        tm = pd.DataFrame(
            {
                "season": df[season_col],
                "home_goals": df[home_col],
                "away_goals": df[away_col],
            }
        )
        frames.append(_add_match_outcome_columns(tm))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# League balance
# ---------------------------------------------------------------------------
def compute_league_balance(season: str, final_table: pd.DataFrame | None, tracking: pd.DataFrame | None) -> dict[str, Any]:
    gini_df = load_processed_csv("gini_by_season.csv")
    points_gini = np.nan
    if gini_df is not None:
        row = gini_df[gini_df["season"] == season]
        if not row.empty:
            points_gini = float(row.iloc[0]["Gini"])

    if final_table is not None and "points" in final_table.columns:
        ft = final_table.sort_values("rank" if "rank" in final_table.columns else "points")
        pts = ft["points"].astype(float).values
    elif tracking is not None:
        tables = build_round_tables(tracking)
        if not tables:
            return {}
        last = tables[max(tables)]
        pts = last.sort_values("position")["points"].astype(float).values
    else:
        return {}

    std = float(np.std(pts, ddof=0))
    top_bottom_gap = float(pts.max() - pts.min()) if len(pts) >= 2 else np.nan

    sorted_pts = np.sort(pts)[::-1]
    gap_1st_6th = float(sorted_pts[0] - sorted_pts[5]) if len(sorted_pts) >= 6 else np.nan
    n_spots = relegation_spots(len(pts))
    rel_line_idx = len(sorted_pts) - n_spots - 1
    gap_6th_rel = float(sorted_pts[5] - sorted_pts[rel_line_idx]) if len(sorted_pts) > 6 and rel_line_idx >= 0 else np.nan

    if pd.isna(points_gini):
        points_gini = calculate_gini(pts)

    gap_score = 1 - min(top_bottom_gap / 80, 1) if pd.notna(top_bottom_gap) else None
    std_score = 1 - min(std / 25, 1)
    gini_score = 1 - min(points_gini / 0.35, 1) if pd.notna(points_gini) else None
    balance = mean_valid_score([gap_score, std_score, gini_score])

    return {
        "final_points_std": std,
        "top_bottom_points_gap": top_bottom_gap,
        "points_gap_1st_6th": gap_1st_6th,
        "points_gap_6th_relegation_line": gap_6th_rel,
        "points_gini": points_gini,
        "league_balance_score": balance,
        "gini_source": "gini_by_season.csv" if gini_df is not None else "recomputed",
    }


# ---------------------------------------------------------------------------
# Attendance (match-level regular + playoff)
# ---------------------------------------------------------------------------
def attendance_placeholder_row(
    *,
    notes: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for col in ATTENDANCE_PLACEHOLDER_COLUMNS:
        if col.endswith("_status"):
            row[col] = status or ATTENDANCE_STATUS
        elif col == "attendance_notes":
            row[col] = notes or ATTENDANCE_NOTES
        elif col == "attendance_available":
            row[col] = False
        else:
            row[col] = np.nan
    return row


def _attendance_season_slug(season: str) -> str:
    return season.replace("/", "_")


def _regular_attendance_paths(season: str) -> list[Path]:
    slug = _attendance_season_slug(season)
    paths = [ATTENDANCE_DIR / f"ligat_haal_{slug}_attendance.csv"]
    if season == "2024/25":
        paths.insert(0, ATTENDANCE_DIR / "ligat_haal_2024_2025_attendance.csv")
    return paths


def load_season_attendance_matches(season: str) -> pd.DataFrame | None:
    """Per-match attendance for one season (regular CSV + playoffs CSV when present)."""
    frames: list[pd.DataFrame] = []
    for path in _regular_attendance_paths(season):
        if path.exists():
            frames.append(pd.read_csv(path))
            break

    playoff_path = ATTENDANCE_DIR / f"ligat_haal_{_attendance_season_slug(season)}_playoffs_attendance.csv"
    if playoff_path.exists():
        frames.append(pd.read_csv(playoff_path))

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    season_col = "Season" if "Season" in df.columns else "season"
    df["season"] = df[season_col]
    df["attendance"] = pd.to_numeric(df["Attendance"], errors="coerce")
    df["match_date"] = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")
    if df["match_date"].isna().all():
        df["match_date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["is_saturday"] = df["match_date"].dt.dayofweek == 5
    df["day_of_week_en"] = df["match_date"].dt.dayofweek.map(
        lambda d: WEEKDAY_ORDER_EN[int(d)] if pd.notna(d) and 0 <= int(d) <= 6 else np.nan
    )
    return df


def load_attendance_by_season(seasons: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for season in seasons:
        df = load_season_attendance_matches(season)
        if df is not None:
            out[season] = df
    return out


def _saturday_weekday_status(
    saturday_avg: float,
    weekday_avg: float,
    pvalue: float,
) -> str:
    diff = saturday_avg - weekday_avg
    if pvalue < 0.05:
        return "computed_saturday_higher" if diff > 0 else "computed_weekday_higher"
    return "computed_no_significant_difference"


def compute_attendance_metrics(season: str, att: pd.DataFrame | None) -> dict[str, Any]:
    """Fill attendance columns when coverage >= ATTENDANCE_MIN_COVERAGE_PCT; else leave empty."""
    if att is None or att.empty:
        return attendance_placeholder_row(notes="No attendance match file found")

    total = len(att)
    valid = att.loc[att["attendance"] > 0].copy()
    valid_count = len(valid)
    coverage = 100.0 * valid_count / total if total else 0.0

    if coverage < ATTENDANCE_MIN_COVERAGE_PCT:
        return attendance_placeholder_row(
            notes=(
                f"Coverage {coverage:.1f}% below minimum "
                f"{ATTENDANCE_MIN_COVERAGE_PCT:.0f}% — metrics left empty"
            ),
            status="insufficient_coverage",
        )

    row = attendance_placeholder_row(
        notes="Match-level attendance (regular + playoff); valid rows have Attendance > 0",
        status="computed",
    )
    row.update({
        "attendance_available": True,
        "attendance_total_match_count": total,
        "attendance_valid_match_count": valid_count,
        "attendance_coverage_percentage": coverage,
        "average_match_attendance_valid": float(valid["attendance"].mean()),
        "median_match_attendance_valid": float(valid["attendance"].median()),
    })

    sat = valid.loc[valid["is_saturday"], "attendance"]
    wkd = valid.loc[~valid["is_saturday"], "attendance"]
    row["attendance_saturday_match_count"] = int(len(sat))
    row["attendance_weekday_match_count"] = int(len(wkd))

    if len(sat) < ATTENDANCE_MIN_DAY_SAMPLES or len(wkd) < ATTENDANCE_MIN_DAY_SAMPLES:
        row["attendance_saturday_vs_weekday_status"] = "insufficient_day_samples"
        row["attendance_notes"] += "; Saturday/weekday comparison needs more valid matches per day"
        return row

    sat_avg = float(sat.mean())
    wkd_avg = float(wkd.mean())
    _, pvalue = ttest_ind(sat, wkd, equal_var=False)

    row.update({
        "attendance_avg_saturday": sat_avg,
        "attendance_avg_weekday": wkd_avg,
        "attendance_saturday_weekday_diff": sat_avg - wkd_avg,
        "attendance_saturday_weekday_ttest_pvalue": float(pvalue),
        "attendance_saturday_vs_weekday_status": _saturday_weekday_status(sat_avg, wkd_avg, pvalue),
    })
    return row


def _attendance_pool_for_chart(summary: pd.DataFrame) -> pd.DataFrame:
    """Valid match attendances from seasons with sufficient coverage."""
    usable = summary[summary["attendance_available"].fillna(False)]
    frames: list[pd.DataFrame] = []
    for season in usable["season"]:
        df = load_season_attendance_matches(season)
        if df is None:
            continue
        valid = df.loc[(df["attendance"] > 0) & df["day_of_week_en"].notna()].copy()
        if not valid.empty:
            frames.append(valid)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_attendance_by_weekday_chart(summary: pd.DataFrame) -> str | None:
    """
    Bar charts: average attendance by weekday and relative index (100 = overall mean).
    Uses seasons with attendance_available (coverage >= ATTENDANCE_MIN_COVERAGE_PCT).
    """
    pool = _attendance_pool_for_chart(summary)
    if pool.empty:
        return None

    by_day = (
        pool.groupby("day_of_week_en", observed=True)
        .agg(
            avg_attendance=("attendance", "mean"),
            match_count=("attendance", "count"),
            total_attendance=("attendance", "sum"),
        )
        .reindex(WEEKDAY_ORDER_EN)
        .dropna(subset=["avg_attendance"])
    )
    if by_day.empty:
        return None

    overall_mean = float(pool["attendance"].mean())
    total_spectators = float(pool["attendance"].sum())
    by_day["relative_index"] = by_day["avg_attendance"] / overall_mean * 100
    by_day["share_of_total_pct"] = by_day["total_attendance"] / total_spectators * 100

    n_seasons = int(summary["attendance_available"].fillna(False).sum())
    colors = ["#4c78a8"] * len(by_day)
    max_day = by_day["avg_attendance"].idxmax()
    if max_day in by_day.index:
        colors[list(by_day.index).index(max_day)] = "#e45756"

    fig, (ax_avg, ax_rel) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    x = np.arange(len(by_day))
    ax_avg.bar(x, by_day["avg_attendance"], color=colors, alpha=0.9)
    ax_avg.axhline(overall_mean, color="gray", linestyle="--", linewidth=1, label="Overall mean")
    ax_avg.set_xticks(x)
    ax_avg.set_xticklabels(by_day.index, rotation=30, ha="right")
    ax_avg.set_ylabel("Average attendance per match")
    ax_avg.set_title("Average Match Attendance by Day of Week")
    ax_avg.legend(loc="upper right", fontsize=9)
    for i, (avg, n) in enumerate(zip(by_day["avg_attendance"], by_day["match_count"])):
        ax_avg.text(i, avg, f"{avg:,.0f}\n(n={int(n)})", ha="center", va="bottom", fontsize=8)

    rel_colors = ["#72b7b2"] * len(by_day)
    max_rel_day = by_day["relative_index"].idxmax()
    if max_rel_day in by_day.index:
        rel_colors[list(by_day.index).index(max_rel_day)] = "#e45756"
    ax_rel.bar(x, by_day["relative_index"], color=rel_colors, alpha=0.9)
    ax_rel.axhline(100, color="gray", linestyle="--", linewidth=1, label="League average (=100)")
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(by_day.index, rotation=30, ha="right")
    ax_rel.set_ylabel("Relative index (overall mean = 100)")
    ax_rel.set_title("Attendance Relative to Overall Average")
    ax_rel.legend(loc="upper right", fontsize=9)
    for i, (idx, share) in enumerate(zip(by_day["relative_index"], by_day["share_of_total_pct"])):
        ax_rel.text(i, idx, f"{idx:.0f}\n({share:.1f}% of total)", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Ligat Ha'al Attendance by Day of Week — {n_seasons} seasons "
        f"(coverage >= {ATTENDANCE_MIN_COVERAGE_PCT:.0f}%, regular + playoff)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_path = CHARTS / "attendance_by_day_of_week.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    audit_path = OUTPUT / "attendance_by_day_of_week_summary.csv"
    by_day.reset_index().to_csv(audit_path, index=False, encoding="utf-8-sig")
    return str(out_path)


# ---------------------------------------------------------------------------
# Data audit
# ---------------------------------------------------------------------------
def build_data_audit(matches: pd.DataFrame) -> pd.DataFrame:
    datasets = [
        ("matches_dated", DATA / "matches" / "matches_all_seasons_ligat_haal_transfermarkt_dated.csv", "match-level",
         "1,2,10,11,13", "home advantage, goals, playoff split"),
        ("matches_tm", DATA / "matches" / "matches_all_seasons_ligat_haal_transfermarkt.csv", "match-level",
         "1,2,10,11", "home advantage, goals"),
        ("regular_season_tracking", DATA / "regular_season_tracking", "team-round-level",
         "1,3,4,5,6,7,8,9,12,13", "title race, relegation, league balance"),
        ("regular_final_tables", DATA / "interim/scraped_standings/regular_final_tables", "team-season-level",
         "10,12", "champion, league balance"),
        ("title_race_season_summary", DATA / "processed/title_race_analysis/season_summary_statistics.csv",
         "processed-analysis-output", "1,2,13", "title race (partial reuse)"),
        ("rounds_led", DATA / "processed/title_race_analysis/rounds_led_by_team_per_season.csv",
         "processed-analysis-output", "5", "champion stability (reference)"),
        ("relegation_summary", DATA / "processed/relegation_competitiveness/summary_all_metrics_by_season.csv",
         "processed-analysis-output", "6,7,8,9", "relegation (partial reuse)"),
        ("relegation_volatility", DATA / "processed/relegation_competitiveness/metric4_relegation_zone_volatility.csv",
         "processed-analysis-output", "7", "relegation volatility (validated)"),
        ("gini_by_season", DATA / "processed/gini_by_season.csv", "processed-analysis-output",
         "12", "league balance"),
        ("playoff_betexplorer", DATA / "playoffs/scraped_betexplorer", "match-level",
         "13", "playoff vs regular"),
        ("attendance_match_files", ATTENDANCE_DIR, "match-level",
         "14,16", "regular + playoff per-match attendance"),
    ]

    rows = []
    for name, path, grain, questions, support in datasets:
        if path.is_dir():
            files = list(path.glob("*.csv"))
            row_count = sum(len(pd.read_csv(f)) for f in files[:50]) if files else 0
            seasons = "multiple"
            columns = "varies"
            exists = bool(files)
            path_str = str(path.relative_to(ROOT))
        else:
            exists = path.exists()
            path_str = str(path.relative_to(ROOT)) if exists else str(path)
            row_count = len(pd.read_csv(path)) if exists else 0
            columns = ",".join(pd.read_csv(path, nrows=0).columns) if exists else ""
            seasons = ",".join(sorted(matches["season"].unique(), key=season_sort_key)) if exists and name.startswith("matches") else "see files"

        rows.append({
            "dataset": name,
            "path": path_str,
            "exists": exists,
            "grain": grain,
            "row_count": row_count,
            "columns": columns,
            "available_seasons": seasons,
            "research_questions_supported": support,
            "reused_in_final_table": name in {
                "matches_dated", "regular_season_tracking", "regular_final_tables",
                "gini_by_season", "playoff_betexplorer", "relegation_volatility",
            },
        })

    # Field checklist
    checklist_fields = {
        "season": matches["season"].notna().any(),
        "round_matchweek": "round" in matches.columns,
        "match_date": "match_date" in matches.columns,
        "home_team": "home" in matches.columns,
        "away_team": "away" in matches.columns,
        "score": "score" in matches.columns,
        "home_goals": False,
        "away_goals": False,
        "match_result": "score" in matches.columns,
        "league_table_by_round": tracking_path("2006/07") is not None,
        "regular_playoff_stage": True,
        "final_standings": (DATA / "interim/scraped_standings/regular_final_tables").exists(),
        "attendance_crowd": ATTENDANCE_DIR.exists(),
        "day_of_week": "day_of_week_en" in matches.columns,
    }
    for field, avail in checklist_fields.items():
        status = "available" if avail else "missing"
        if field in ("attendance_crowd", "day_of_week"):
            status = "placeholder_for_future_stage" if field == "attendance_crowd" else (
                "available_not_used" if avail else "missing"
            )
        rows.append({
            "dataset": f"field_checklist:{field}",
            "path": "matches + tracking",
            "exists": avail,
            "grain": "field",
            "row_count": "",
            "columns": field,
            "available_seasons": "",
            "research_questions_supported": status,
            "reused_in_final_table": field not in ("attendance_crowd",),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feasibility table
# ---------------------------------------------------------------------------
def build_feasibility_table(summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Research question feasibility; extended when advanced features are present."""
    att_ready = 0
    if summary is not None and "attendance_available" in summary.columns:
        att_ready = int(summary["attendance_available"].fillna(False).sum())

    att_q14_status = "answerable_now" if att_ready else ATTENDANCE_STATUS
    att_q14_expl = (
        f"Saturday vs non-Saturday from match attendance ({att_ready} seasons with "
        f"coverage >= {ATTENDANCE_MIN_COVERAGE_PCT:.0f}%)."
        if att_ready
        else ATTENDANCE_NOTES
    )
    att_q16_status = att_q14_status
    att_q16_expl = (
        "average_match_attendance_valid filled for seasons with sufficient attendance coverage."
        if att_ready
        else ATTENDANCE_NOTES
    )

    questions = [
        (1, "Which seasons had the closest title race?", "title_race_closeness_score", "answerable_now",
         "Computed from tracking; final_gap from playoff final table in playoff seasons.", "Review top seasons in charts.",
         "playoff_final_table + tracking"),
        (2, "In which round was the championship mathematically decided?", "title_decision_round", "answerable_now",
         "gap > remaining_rounds*3, or final round (ties resolved by tiebreaker).", "Verify in title_decision_audit.csv.", "title_decision_audit"),
        (3, "How many teams were still in the title race near the end?", "teams_in_title_race_5_rounds_before_end",
         "answerable_now", "Teams within 7 points of leader at checkpoint (5 rounds before end).", "Compare across seasons.", "recomputed_from_tracking"),
        (4, "How many lead changes occurred after round 10?", "lead_changes_after_round_10", "answerable_now",
         "Leader changes after global round 10; includes championship playoff when tracking available.",
         "See first_round_champion_distribution.", "regular + championship_playoff_tracking"),
        (5, "How stable was the champion throughout the season?", "champion_rounds_as_leader_percentage",
         "answerable_now",
         "Rounds as leader vs eventual champion; full title timeline in playoff seasons.",
         "Cross-check rounds_led CSV.", "regular + championship_playoff_tracking"),
        (6, "How many teams were fighting relegation near the end?", "teams_in_relegation_race_last_5_rounds",
         "answerable_now",
         "Survival line = first relegation spot; abs(points-survival)<=6, rank_snap>=N-6; full-season incl. playoff.",
         "See relegation_race_audit.csv; compare regular-only columns.", "relegation_playoff_tracking + tracking"),
        (7, "How volatile was the relegation zone?", "relegation_zone_changes_total", "answerable_now",
         "Notebook M4: half symmetric-diff zone changes on regular + relegation playoff timeline.", "Use charts by season.", "metric4_notebook_logic"),
        (8, "In which round was relegation mathematically decided?", "relegation_decision_round", "answerable_now",
         "From notebook Metric 2 (metric2_relegation_decision_timing.csv).", "Verify per-team rows in metric2 CSV.", "metric2_notebook"),
        (9, "How close was the relegation battle?", "relegation_closeness_score", "answerable_now",
         "Composite score; late_contenders uses full-season teams_in_relegation_race_last_5_rounds.", "Report top dramatic seasons.", "recomputed"),
        (10, "Is there home advantage in Ligat Ha'Al?", "home_win_percentage", "answerable_now",
         "From dated match scores.", "Pool across seasons in report.", "matches_all_seasons_dated"),
        (11, "Which seasons were more attacking?", "average_goals_per_match", "answerable_now",
         "From valid match scores.", "Compare 0-0 rate and over 2.5.", "matches_all_seasons_dated"),
        (12, "How balanced was the league overall?", "league_balance_score", "answerable_now",
         "Std, gap, Gini from final table; Gini reused from gini_by_season.csv.", "Highlight least balanced seasons.", "gini_by_season.csv + final tables"),
        (13, "How different were playoffs compared to the regular season?", "regular_average_goals_per_match",
         "answerable_now", "Regular from TM matches; playoff from betexplorer when needed.", "Mark insufficient_stage_data seasons.", "matches + betexplorer playoffs"),
        (14, "Does Saturday attendance differ from non-Saturday attendance?", "attendance_avg_saturday",
         att_q14_status, att_q14_expl, "Compare attendance_saturday_weekday_diff across seasons.",
         "notebooks/attendance match files"),
        (15, "Does weekday affect attendance?", "attendance_by_weekday_anova_pvalue", ATTENDANCE_STATUS,
         ATTENDANCE_NOTES, "Weekday ANOVA not yet implemented.", "placeholder"),
        (16, "Do competitive seasons attract higher attendance?", "average_match_attendance_valid",
         att_q16_status, att_q16_expl, "Correlate with title_race_closeness_score.", "notebooks/attendance match files"),
        (17, "Do top-of-table matches attract higher attendance?", "top_match_attendance_avg", ATTENDANCE_STATUS,
         ATTENDANCE_NOTES, "Define top-match threshold then compute.", "placeholder"),
    ]

    adv_status = "answerable_now"
    if summary is not None and "combined_competitiveness_score" in summary.columns:
        n_adv = int(summary["combined_competitiveness_score"].notna().sum())
        adv_expl = f"Combined index from title, relegation, and balance scores ({n_adv} seasons)."
    else:
        adv_status = "placeholder_pending_advanced_features"
        adv_expl = "Run advanced_competitiveness_features module."

    questions.extend(
        [
            (18, "How concentrated is scoring among star players?", "offensive_concentration_gini", adv_status,
             "Gini and top-3 share from Transfermarkt scorers.", "Review offensive_concentration_audit.csv.",
             "player_stats scorers"),
            (19, "Does star concentration relate to league balance?", "share_of_league_goals_by_top_3_scorers", adv_status,
             "Compare share_of_league_goals_by_top_3_scorers vs league_balance_score.", "Scatter in report.",
             "offensive_concentration + league_balance"),
            (20, "Are relegation battles more volatile than title races?", "relegation_vs_title_volatility_ratio", adv_status,
             "Late-season rank movement in top 4 vs bottom 4 zones.", "See rank_dynamics_audit.csv.",
             "rank_dynamics export"),
            (21, "Did playoffs increase late-season rank movement?", "late_season_rank_volatility", adv_status,
             "Compare across pre/post-2009/10 eras.", "Era comparison in discussion.",
             "rank_dynamics export"),
            (22, "Is the relegation playoff more impactful than the championship playoff?", "playoff_impact_balance", adv_status,
             "Descriptive comparison of zone/survival flips vs leadership changes.", "playoff_impact_comparison_audit.csv",
             "championship + relegation playoff tracking"),
            (23, "Which season was most competitive overall?", "combined_competitiveness_score", adv_expl if adv_status == "answerable_now" else adv_status,
             adv_expl if adv_status == "answerable_now" else "Combined normalized index.",
             "combined_competitiveness_ranking.csv", "title + relegation + balance scores"),
            (24, "What are data-quality limitations per season?", "combined_data_quality_score", adv_status,
             "Weighted coverage across attendance, player stats, playoffs, rank dynamics.", "data_quality_summary.csv",
             "multi-source flags"),
        ]
    )

    rows = []
    for qid, rq, feat, status, expl, next_step, source in questions:
        rows.append({
            "question_id": qid,
            "research_question": rq,
            "feature_name": feat,
            "required_columns": "see season_research_summary.csv",
            "available_columns": feat,
            "status": status,
            "explanation": expl,
            "recommended_next_step": next_step,
            "source_used": source,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _seasons_for_charts(df: pd.DataFrame, completed_only: bool = True) -> list[str]:
    sub = df[df["is_completed_season"]] if completed_only and "is_completed_season" in df.columns else df
    return sorted(sub["season"].tolist(), key=season_sort_key)


def save_bar_chart(df: pd.DataFrame, col: str, title: str, fname: str, completed_only: bool = True) -> None:
    sub = df[df["is_completed_season"]] if completed_only else df
    seasons = _seasons_for_charts(df, completed_only)
    vals = [sub.loc[sub["season"] == s, col].iloc[0] if s in sub["season"].values else np.nan for s in seasons]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(seasons, vals, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Season")
    ax.set_ylabel(col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS / fname, dpi=150)
    plt.close(fig)


def save_stacked_hda(df: pd.DataFrame) -> None:
    sub = df[df["is_completed_season"]]
    seasons = _seasons_for_charts(df)
    home = [sub.loc[sub["season"] == s, "home_win_percentage"].iloc[0] for s in seasons]
    draw = [sub.loc[sub["season"] == s, "draw_percentage"].iloc[0] for s in seasons]
    away = [sub.loc[sub["season"] == s, "away_win_percentage"].iloc[0] for s in seasons]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(seasons, home, label="Home win %")
    ax.bar(seasons, draw, bottom=home, label="Draw %")
    ax.bar(seasons, away, bottom=np.array(home) + np.array(draw), label="Away win %")
    ax.set_title("Home / Draw / Away Result Percentages by Season")
    ax.set_ylabel("Percentage")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS / "home_draw_away_percentages_by_season.png", dpi=150)
    plt.close(fig)


def save_playoff_comparison(df: pd.DataFrame, col_reg: str, col_po: str, title: str, fname: str) -> None:
    sub = df[(df["is_completed_season"]) & (df["playoff_vs_regular_competitiveness_status"] == "computed")]
    if sub.empty:
        return
    seasons = sorted(sub["season"].tolist(), key=season_sort_key)
    reg = sub.set_index("season").loc[seasons, col_reg].values
    po = sub.set_index("season").loc[seasons, col_po].values
    x = np.arange(len(seasons))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, reg, w, label="Regular")
    ax.bar(x + w / 2, po, w, label="Playoff")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHARTS / fname, dpi=150)
    plt.close(fig)


def save_first_round_distribution(dist_df: pd.DataFrame) -> None:
    counts = dist_df["first_round_champion_was_1st"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [str(k) for k in counts.index]
    ax.bar(labels, counts.values, color=["coral", "seagreen"][: len(labels)])
    ax.set_title("Distribution: First-Round Leader = Eventual Champion?")
    ax.set_xlabel("first_round_champion_was_1st")
    ax.set_ylabel("Number of seasons")
    plt.tight_layout()
    fig.savefig(CHARTS / "first_round_champion_was_1st_distribution.png", dpi=150)
    plt.close(fig)


def save_title_race_timeline_chart(
    audit: pd.DataFrame,
    season: str,
    summary_row: pd.Series,
    fname: str,
    competitiveness_label: str,
) -> None:
    """Round-by-round title race timeline for one season (points + gap)."""
    sub = audit[audit["season"] == season].sort_values("global_round").copy()
    if sub.empty:
        return

    regular_end = int(sub.loc[sub["stage"] == "regular", "global_round"].max())
    closeness = summary_row.get("title_race_closeness_score", np.nan)
    champion = summary_row.get("champion", "")
    runner_up = summary_row.get("runner_up", "")
    final_gap = summary_row.get("final_gap_1st_2nd", np.nan)
    lead_changes = summary_row.get("lead_changes_after_round_10", np.nan)
    decision_round = summary_row.get("title_decision_round", np.nan)

    rounds = sub["global_round"].astype(int)
    leader_changed = sub["leader"] != sub["leader"].shift(1)
    change_rounds = sub.loc[leader_changed & (sub["global_round"] > 1), "global_round"]

    fig, (ax_pts, ax_gap) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    ax_pts.plot(rounds, sub["leader_points"], color="#1f77b4", linewidth=2.2, label="Leader points", marker="o", markersize=4)
    ax_pts.plot(
        rounds, sub["runner_up_points"], color="#ff7f0e", linewidth=2.2,
        label="Runner-up points", marker="s", markersize=4,
    )
    ax_pts.fill_between(rounds, sub["leader_points"], sub["runner_up_points"], alpha=0.12, color="#1f77b4")

    if sub["stage"].eq("championship_playoff").any():
        playoff_start = int(sub.loc[sub["stage"] == "championship_playoff", "global_round"].min())
        ax_pts.axvspan(playoff_start - 0.5, rounds.max() + 0.5, alpha=0.08, color="purple", label="Championship playoff")
        ax_gap.axvspan(playoff_start - 0.5, rounds.max() + 0.5, alpha=0.08, color="purple")

    ax_pts.axvline(regular_end + 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Regular season end")

    for rnd in change_rounds:
        ax_pts.axvline(rnd, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.55)
        ax_gap.axvline(rnd, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.55)

    if pd.notna(decision_round):
        ax_pts.axvline(decision_round, color="#d62728", linestyle="-.", linewidth=1.2, alpha=0.8, label="Title decided")
        ax_gap.axvline(decision_round, color="#d62728", linestyle="-.", linewidth=1.2, alpha=0.8)

    ax_pts.set_ylabel("Points")
    ax_pts.legend(loc="upper left", fontsize=9)
    ax_pts.grid(True, alpha=0.25)

    ax_gap.bar(rounds, sub["gap_1st_2nd"], color="#6baed6", alpha=0.85, width=0.85, label="Gap 1st–2nd")
    ax_gap.set_ylabel("Point gap")
    ax_gap.set_xlabel("Round (global timeline: regular + championship playoff)")
    ax_gap.grid(True, axis="y", alpha=0.25)

    title = (
        f"Title Race Timeline — {season} ({competitiveness_label})\n"
        f"Closeness score: {closeness:.1f} | Champion: {champion} | "
        f"Runner-up: {runner_up} | Final gap: {final_gap:.0f} pts | "
        f"Lead changes after R10: {int(lead_changes) if pd.notna(lead_changes) else '—'}"
    )
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_title_race_timeline_comparison_charts(summary: pd.DataFrame) -> list[str]:
    """Most vs least competitive title-race seasons (by title_race_closeness_score)."""
    audit_path = OUTPUT / "title_decision_audit.csv"
    if not audit_path.exists():
        return []

    audit = pd.read_csv(audit_path)
    completed = summary[summary["is_completed_season"]].dropna(subset=["title_race_closeness_score"])
    if completed.empty:
        return []

    most_row = completed.loc[completed["title_race_closeness_score"].idxmax()]
    least_row = completed.loc[completed["title_race_closeness_score"].idxmin()]

    jobs = [
        (most_row, "most_competitive", "Most competitive title race"),
        (least_row, "least_competitive", "Least competitive title race"),
    ]
    created: list[str] = []
    for row, slug, label in jobs:
        season = row["season"]
        season_slug = season.replace("/", "_")
        out_name = f"title_race_timeline_{slug}_{season_slug}.png"
        save_title_race_timeline_chart(audit, season, row, out_name, label)
        created.append(str(CHARTS / out_name))
    return created


def create_charts(summary: pd.DataFrame, dist_df: pd.DataFrame) -> list[str]:
    CHARTS.mkdir(parents=True, exist_ok=True)
    chart_jobs = [
        ("final_gap_1st_2nd", "Final Gap 1st–2nd by Season", "final_gap_1st_2nd_by_season.png"),
        ("title_decision_round", "Title Decision Round by Season", "title_decision_round_by_season.png"),
        ("lead_changes_after_round_10", "Lead Changes After Round 10", "lead_changes_after_round_10_by_season.png"),
        ("teams_in_title_race_5_rounds_before_end", "Teams in Title Race (5 Rounds Before End)",
         "teams_in_title_race_5_rounds_before_end.png"),
        ("champion_rounds_as_leader_percentage", "Champion Leadership % by Season",
         "champion_leadership_percentage_by_season.png"),
        ("title_race_closeness_score", "Title Race Closeness Score", "title_race_closeness_score_by_season.png"),
        ("relegation_zone_changes_total", "Relegation Zone Changes by Season", "relegation_zone_changes_by_season.png"),
        ("teams_in_relegation_race_last_5_rounds", "Teams in Relegation Race (Last 5 Rounds)",
         "teams_in_relegation_race_last_5_rounds.png"),
        ("relegation_closeness_score", "Relegation Closeness Score", "relegation_closeness_score_by_season.png"),
        ("average_goals_per_match", "Average Goals per Match by Season", "average_goals_per_match_by_season.png"),
        ("league_balance_score", "League Balance Score by Season", "league_balance_score_by_season.png"),
    ]
    created = []
    for col, title, fname in chart_jobs:
        save_bar_chart(summary, col, title, fname)
        created.append(str(CHARTS / fname))
    save_stacked_hda(summary)
    created.append(str(CHARTS / "home_draw_away_percentages_by_season.png"))
    save_first_round_distribution(dist_df)
    created.append(str(CHARTS / "first_round_champion_was_1st_distribution.png"))
    save_playoff_comparison(
        summary, "regular_average_goals_per_match", "playoff_average_goals_per_match",
        "Regular vs Playoff Average Goals", "regular_vs_playoff_goals.png",
    )
    save_playoff_comparison(
        summary, "regular_home_win_percentage", "playoff_home_win_percentage",
        "Regular vs Playoff Home Win %", "regular_vs_playoff_home_win_percentage.png",
    )
    for p in ["regular_vs_playoff_goals.png", "regular_vs_playoff_home_win_percentage.png"]:
        if (CHARTS / p).exists():
            created.append(str(CHARTS / p))
    created.extend(save_title_race_timeline_comparison_charts(summary))
    weekday_chart = save_attendance_by_weekday_chart(summary)
    if weekday_chart:
        created.append(weekday_chart)
    return created


# ---------------------------------------------------------------------------
# Hebrew findings (template filled after compute)
# ---------------------------------------------------------------------------
def write_hebrew_findings(summary: pd.DataFrame, feasibility: pd.DataFrame) -> None:
    completed = summary[summary["is_completed_season"]]
    top_title = completed.nlargest(5, "title_race_closeness_score")[["season", "title_race_closeness_score", "champion"]]
    top_rel = completed.nlargest(5, "relegation_closeness_score")[["season", "relegation_closeness_score"]]
    top_bal = completed.nlargest(5, "league_balance_score")[["season", "league_balance_score"]]
    warn_seasons = completed[completed["data_quality_flag"].astype(str).str.contains("backup|incomplete|missing", case=False, na=False)]

    lines = [
        "# ממצאי מחקר – טבלת סיכום עונתית (שלב ראשון)",
        "",
        "## 1. מטרת הטבלה המחקרית",
        "בניית טבלת סיכום ברמת עונה שמרכזת את כל שאלות המחקר של הפרויקט: תחרותיות, מאבק אליפות, מאבק ירידה, יתרון ביתיות, שערים, איזון ליגה ופלייאוף.",
        "",
        "## 2. מחברות וקבצים קיימים שנוצלו",
        "- `matches_all_seasons_ligat_haal_transfermarkt_dated.csv` – שערים ויתרון ביתיות",
        "- `regular_season_tracking_*.csv` – מדדי מאבק אליפות וירידה",
        "- `regular_final_table_*.csv` – אלופה, מרחק נקודות, איזון",
        "- `gini_by_season.csv` – מקדם ג'יני (נבדק מול חישוב מחדש)",
        "- `metric4_relegation_zone_volatility.csv` – אימות תנודתיות אזור ירידה",
        "- `playoffs/scraped_betexplorer/` – השוואת פלייאוף מול עונה רגילה",
        "- `season_summary_statistics.csv` – עזר לזיהוי מבנה פלייאוף (לא לשינוי מנהיגות אחרי סיבוב 10)",
        "",
        "## 3. שאלות מחקר שנענו כבר עכשיו",
        "שאלות 1–13 (תחרותיות, אליפות, ירידה, ביתיות, שערים, איזון, פלייאוף) – ראו `research_question_feasibility.csv` עם סטטוס `answerable_now`.",
        "",
        "## 4. נתוני קהל",
        f"עמודות קהל מולאו לעונות עם כיסוי >= {ATTENDANCE_MIN_COVERAGE_PCT:.0f}% (משחקי עונה רגילה + פלייאוף). עונות עם כיסוי נמוך נשארו ריקות.",
        "",
        "## 5. ממצאים ראשוניים – מאבק אליפות",
    ]
    for _, r in top_title.iterrows():
        lines.append(f"- {r['season']}: ציון קרבות {r['title_race_closeness_score']:.1f} (אלופה: {r['champion']})")
    lines += [
        "",
        "**נוסחת title_race_closeness_score:** ממוצע של gap_score, decision_score, race_teams_score, lead_changes_score × 100.",
        "",
        "## 6. ממצאים ראשוניים – מאבק ירידה",
    ]
    for _, r in top_rel.iterrows():
        lines.append(f"- {r['season']}: ציון קרבות ירידה {r['relegation_closeness_score']:.1f}")
    lines += [
        "",
        "**מדדי מאבק ירידה – שני סוגים:**",
        "- `teams_in_relegation_race_last_*_regular_rounds` – נקודת ביקורת בעונה הרגילה בלבד (5/6/8 סיבובים לפני סוף העונה הרגילה).",
        "- `teams_in_relegation_race_last_*_rounds` – נקודת ביקורת על ציר הירידה המלא: עונה רגילה + פלייאוף ירידה (כשקיים). בעונות ללא פלייאוף הערכים זהים.",
        "- כלל: קבוצה במאבק אם `abs(נקודות - קו_הישרדות) <= 6` **וגם** `rank_snap >= N - 6` (בלוק תחתון בלבד).",
        "- קו הישרדות: נקודות הקבוצה **במקום הירידה הראשון** (11 בליגת 12, 13 ב-14, 15 ב-16; בפלייאוף ירידה — לפי גודל הטבלה).",
        "- נקודת ביקורת מלאה: 5/6/8 סיבובים לפני סוף ציר הירידה (כולל פלייאוף ירידה כשקיים).",
        "- `relegation_zone_changes_total`: כמו Metric 4 — שינויי אזור ירידה (חצי symmetric diff) על ציר מלא.",
        "- `relegation_race_method`: `full_season_no_playoff` | `full_season_with_relegation_playoff` | `regular_only_fallback` | `unavailable`.",
        "- `relegation_closeness_score` משתמש ב-`teams_in_relegation_race_last_5_rounds` (ציר מלא).",
        "",
        "**נוסחת relegation_closeness_score:** ממוצע של late_contenders (מלא), volatility, decision × 100.",
        "",
        "## 7. יתרון ביתיות ושערים",
        f"- ממוצע ביתיות (כל העונות המושלמות): ניצחון בית {completed['home_win_percentage'].mean():.1f}%, תיקו {completed['draw_percentage'].mean():.1f}%, ניצחון חוץ {completed['away_win_percentage'].mean():.1f}%",
        f"- ממוצע שערים למשחק: {completed['average_goals_per_match'].mean():.2f}",
        "",
        "## 8. איזון הליגה",
    ]
    for _, r in top_bal.iterrows():
        lines.append(f"- {r['season']}: ציון איזון {r['league_balance_score']:.1f}")
    lines += [
        "",
        "## 9. מגבלות נתונים",
        "- `title_decision_round`: gap > remaining_rounds×3, או בסיבוב האחרון (תיקו נפתר בשובר שיוויון); כולל פלייאוף.",
        "- `relegation_decision_round` מגיע מ-Metric 2 בנוטבוק (`metric2_notebook`).",
        "- פלייאוף ירידה: בציר המלא הטבלה עשויה להכיל רק קבוצות פלייאוף תחתון; 2 מקומות ירידה (ברירת מחדל).",
        "- 2024/25: קובץ מעקב גיבוי אם חסר קובץ ראשי.",
        "- 2025/26: נוספה לטבלה לאחר השלמת איסוף משחקים ומעקב מ-Transfermarkt.",
    ]
    if not warn_seasons.empty:
        lines.append("- עונות עם אזהרות: " + ", ".join(warn_seasons["season"].tolist()))
    lines += [
        "",
        "## 10. צעדים הבאים",
        "1. השלמת כיסוי קהל לעונות עם נתונים חלקיים",
        "2. מדד ANOVA לפי יום בשבוע ומשחקי צמרת (שאלות 15, 17)",
        "3. יצירת גרפי קהל ושבת/חול",
        "",
        "## פסקה מוצעת לדוח הסופי",
        "",
        "> בשלב זה נבנתה טבלת סיכום עונתית הכוללת את כלל שאלות המחקר של הפרויקט. מדדי קהל (ממוצע למשחק, שבת מול חול) מולאו לעונות עם כיסוי מספק; עונות עם כיסוי נמוך נותרו ריקות.",
        "",
    ]
    (OUTPUT / "final_research_findings_hebrew.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build_season_summary() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    matches_path = DATA / "matches" / "matches_all_seasons_ligat_haal_transfermarkt_dated.csv"
    matches = pd.read_csv(matches_path)
    matches["home"] = matches["home"].map(normalize_team)
    matches["away"] = matches["away"].map(normalize_team)

    seasons = list_seasons_from_matches(matches)
    attendance_by_season = load_attendance_by_season(seasons)
    warnings_log: list[str] = []
    rows = []
    all_title_audit_rows: list[dict[str, Any]] = []
    all_relegation_audit_rows: list[dict[str, Any]] = []

    for season in seasons:
        is_completed = season_sort_key(season) <= season_sort_key(COMPLETED_SEASON_END)
        tracking, track_flag = load_tracking(season)
        final_table = load_final_table(season)

        notes_parts: list[str] = []
        flags: list[str] = []
        if track_flag == "tracking_backup_2024_25":
            flags.append("tracking_backup_2024_25")
            notes_parts.append("Used regular_season_tracking_2024_25 backup file.")
        if not is_completed:
            flags.append("incomplete_season")
            notes_parts.append("Season not completed; excluded from completed-season rankings.")
        if tracking is None:
            flags.append("missing_tracking")
            notes_parts.append("Missing standings-by-round tracking.")

        row: dict[str, Any] = {
            "season": season,
            "is_completed_season": is_completed,
            "has_playoff_data": (DATA / "playoffs" / "scraped_betexplorer" / "all_championship_betexplorer.csv").exists(),
            "data_quality_flag": "; ".join(flags) if flags else "ok",
        }

        # Title / relegation from tracking
        if tracking is not None:
            title_m = compute_title_race_metrics(season, tracking, final_table)
            rel_m = compute_relegation_metrics(season, tracking)
            row.update({
                k: v for k, v in title_m.items()
                if k not in ("title_notes", "title_audit_rows")
            })
            row.update({k: v for k, v in rel_m.items() if k != "relegation_audit_rows"})
            if rel_m.get("relegation_audit_rows"):
                all_relegation_audit_rows.extend(rel_m["relegation_audit_rows"])
            if title_m.get("title_notes"):
                notes_parts.append(title_m["title_notes"])
            if title_m.get("title_audit_rows"):
                all_title_audit_rows.extend(title_m["title_audit_rows"])
            regular_rounds = title_m.get("regular_rounds", np.nan)
            row["total_rounds_if_available"] = title_m.get(
                "total_rounds_in_title_race", regular_rounds
            )
        else:
            regular_rounds = np.nan
            for col in [
                "champion", "runner_up", "champion_points", "runner_up_points", "final_gap_1st_2nd",
                "title_decision_round", "rounds_before_end_when_title_decided", "title_decision_method",
                "teams_in_title_race_5_rounds_before_end", "teams_in_title_race_6_rounds_before_end",
                "teams_in_title_race_8_rounds_before_end", "average_title_race_teams_last_5_rounds",
                "lead_changes_after_round_10", "first_round_champion_was_1st",
                "number_of_rounds_champion_was_1st", "champion_rounds_as_leader_percentage",
                "longest_consecutive_lead_streak_by_champion", "title_race_closeness_score",
                "teams_in_relegation_race_last_5_regular_rounds",
                "teams_in_relegation_race_last_6_regular_rounds",
                "teams_in_relegation_race_last_8_regular_rounds",
                "teams_in_relegation_race_last_5_rounds", "teams_in_relegation_race_last_6_rounds",
                "teams_in_relegation_race_last_8_rounds", "relegation_race_method",
                "relegation_zone_changes_total",
                "relegation_zone_changes_after_round_10", "relegation_zone_changes_last_5_rounds",
                "relegation_decision_round", "relegation_decision_method", "relegation_closeness_score",
            ]:
                row[col] = np.nan

        # Match metrics
        match_m = compute_match_metrics(season, matches, regular_rounds)
        row.update({k: v for k, v in match_m.items() if k != "match_notes"})
        if match_m.get("match_notes"):
            notes_parts.append(match_m["match_notes"])

        row["regular_rounds"] = regular_rounds
        if "total_rounds_if_available" not in row or pd.isna(row.get("total_rounds_if_available")):
            row["total_rounds_if_available"] = regular_rounds

        # League balance
        bal = compute_league_balance(season, final_table, tracking)
        row.update(bal)

        # Attendance (regular + playoff match files)
        att_m = compute_attendance_metrics(season, attendance_by_season.get(season))
        row.update(att_m)

        row["notes"] = "; ".join(notes_parts) if notes_parts else ""
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Drop internal columns
    drop_cols = [
        c for c in summary.columns
        if c.startswith("relegation_source")
        or c in ("gini_source", "final_gap_source", "total_rounds_in_title_race", "total_rounds_in_relegation_race")
    ]
    summary = summary.drop(columns=[c for c in drop_cols if c in summary.columns])

    if apply_goal_contribution_to_summary is not None:
        summary = apply_goal_contribution_to_summary(summary, seasons=seasons, scrape_if_missing=True)
    else:
        summary["players_with_15_plus_goals_assists"] = np.nan
        summary["distinct_teams_with_15_plus_goals_assists"] = np.nan
        summary["player_stats_available"] = False

    advanced_created: dict[str, list[str]] = {}
    if apply_advanced_features is not None:
        summary, advanced_created = apply_advanced_features(
            summary,
            seasons,
            is_playoff_season_fn=is_playoff_season,
            load_cp_fn=load_championship_playoff_tracking,
            load_rp_fn=load_relegation_playoff_tracking,
        )
    else:
        warnings_log.append("advanced_competitiveness_features module not imported")

    # first_round distribution
    dist = summary[summary["is_completed_season"]][["season", "first_round_champion_was_1st"]].copy()
    dist_counts = dist["first_round_champion_was_1st"].value_counts(dropna=False).reset_index()
    dist_counts.columns = ["first_round_champion_was_1st", "season_count"]
    dist_counts.to_csv(OUTPUT / "first_round_champion_was_1st_distribution.csv", index=False, encoding="utf-8-sig")

    if all_title_audit_rows:
        title_audit_df = pd.DataFrame(all_title_audit_rows)
        title_audit_df.to_csv(OUTPUT / "title_decision_audit.csv", index=False, encoding="utf-8-sig")

    if all_relegation_audit_rows:
        relegation_audit_df = pd.DataFrame(all_relegation_audit_rows)
        relegation_audit_df.to_csv(OUTPUT / "relegation_race_audit.csv", index=False, encoding="utf-8-sig")

    # Final report subset
    report_cols = [
        "season", "champion", "runner_up", "final_gap_1st_2nd", "title_decision_round",
        "rounds_before_end_when_title_decided", "teams_in_title_race_5_rounds_before_end",
        "lead_changes_after_round_10", "champion_rounds_as_leader_percentage",
        "teams_in_relegation_race_last_5_rounds", "relegation_zone_changes_total",
        "home_win_percentage", "draw_percentage", "away_win_percentage", "average_goals_per_match",
        "title_race_closeness_score", "relegation_closeness_score", "league_balance_score",
        "attendance_available", "average_match_attendance_valid", "attendance_avg_saturday",
        "attendance_avg_weekday",         "attendance_saturday_weekday_diff",
        "attendance_saturday_vs_weekday_status",
        "players_with_15_plus_goals_assists",
        "distinct_teams_with_15_plus_goals_assists",
        "top_scorer_goals",
        "share_of_league_goals_by_top_3_scorers",
        "offensive_concentration_gini",
        "avg_rank_changes_per_round",
        "max_single_round_rank_swing",
        "late_season_rank_volatility",
        "title_zone_rank_volatility",
        "relegation_zone_rank_volatility",
        "relegation_vs_title_volatility_ratio",
        "combined_competitiveness_score",
        "combined_competitiveness_rank",
        "championship_playoff_first_place_flipped",
        "relegation_playoff_survival_flips",
        "playoff_impact_balance",
        "attendance_coverage_percentage",
        "player_stats_assists_available",
        "playoff_data_completeness_flag",
        "combined_data_quality_score",
        "data_quality_flag", "notes",
    ]
    final_report = summary[[c for c in report_cols if c in summary.columns]]
    final_report.to_csv(OUTPUT / "final_report_research_table.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(OUTPUT / "season_research_summary.csv", index=False, encoding="utf-8-sig")

    audit = build_data_audit(matches)
    audit.to_csv(OUTPUT / "data_audit_summary.csv", index=False, encoding="utf-8-sig")

    feasibility = build_feasibility_table(summary)
    feasibility.to_csv(OUTPUT / "research_question_feasibility.csv", index=False, encoding="utf-8-sig")

    write_hebrew_findings(summary, feasibility)
    create_charts(summary, dist_counts)

    return summary, feasibility, warnings_log, advanced_created


def print_validation(summary: pd.DataFrame, feasibility: pd.DataFrame) -> None:
    completed = summary[summary["is_completed_season"]]
    full_tracking = completed[~completed["data_quality_flag"].astype(str).str.contains("missing_tracking", na=False)]

    required_files = [
        OUTPUT / "season_research_summary.csv",
        OUTPUT / "final_report_research_table.csv",
        OUTPUT / "research_question_feasibility.csv",
        OUTPUT / "data_audit_summary.csv",
        OUTPUT / "first_round_champion_was_1st_distribution.csv",
        OUTPUT / "title_decision_audit.csv",
        OUTPUT / "relegation_race_audit.csv",
        OUTPUT / "final_research_findings_hebrew.md",
        CHARTS / "title_decision_round_by_season.png",
    ]

    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY BUILD – VALIDATION")
    print("=" * 60)
    print(f"Seasons in season_research_summary.csv: {len(summary)}")
    print(f"Columns: {len(summary.columns)}")
    print(f"Completed seasons: {len(completed)}")
    print(f"Seasons with full standings-by-round data: {len(full_tracking)}")
    print("\nTop 5 title race closeness seasons:")
    print(completed.nlargest(5, "title_race_closeness_score")[["season", "title_race_closeness_score", "champion"]].to_string(index=False))
    print("\nTop 5 relegation closeness seasons:")
    print(completed.nlargest(5, "relegation_closeness_score")[["season", "relegation_closeness_score"]].to_string(index=False))
    print("\nTop 5 most balanced seasons:")
    print(completed.nlargest(5, "league_balance_score")[["season", "league_balance_score"]].to_string(index=False))
    att_ok = completed[completed["attendance_available"].fillna(False)]
    print(f"\nSeasons with attendance metrics: {len(att_ok)} / {len(completed)}")
    if not att_ok.empty:
        print(att_ok[["season", "attendance_coverage_percentage", "average_match_attendance_valid",
                       "attendance_saturday_weekday_diff"]].to_string(index=False))
    low_cov = completed[~completed["attendance_available"].fillna(False)]
    if not low_cov.empty:
        print(f"\nSeasons left empty (low/missing attendance coverage): {', '.join(low_cov['season'].tolist())}")

    if "players_with_15_plus_goals_assists" in completed.columns:
        ga_ok = completed[completed["player_stats_available"].fillna(False)]
        print(f"\nSeasons with 15+ G+A player stats: {len(ga_ok)} / {len(completed)}")
        if not ga_ok.empty:
            print(
                ga_ok[
                    [
                        "season",
                        "players_with_15_plus_goals_assists",
                        "distinct_teams_with_15_plus_goals_assists",
                    ]
                ].to_string(index=False)
            )

    print("\nRequired output files:")
    for p in required_files:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.relative_to(ROOT)}")

    m2_method = completed[completed["relegation_decision_method"] == "metric2_notebook"]
    if not m2_method.empty:
        print(f"\nRelegation decision from notebook Metric 2: {len(m2_method)} seasons")
    warn = summary[summary["data_quality_flag"].astype(str) != "ok"]
    if not warn.empty:
        print("\nSeason warnings:")
        for _, r in warn.iterrows():
            print(f"  {r['season']}: {r['data_quality_flag']} | {r.get('notes', '')[:80]}")


def print_advanced_features_summary(
    summary: pd.DataFrame, advanced_created: dict[str, list[str]]
) -> None:
    print("\n" + "=" * 60)
    print("ADVANCED COMPETITIVENESS FEATURES")
    print("=" * 60)
    groups = ["offensive", "rank_dynamics", "combined", "playoff", "data_quality", "charts"]
    for g in groups:
        files = advanced_created.get(g, [])
        print(f"\n{g}: {len(files)} file(s)")
        for f in files:
            print(f"  - {Path(f).name}")

    if "combined_competitiveness_score" in summary.columns:
        top = summary.dropna(subset=["combined_competitiveness_score"]).nlargest(3, "combined_competitiveness_score")
        if not top.empty:
            print("\nTop 3 combined competitiveness seasons:")
            print(top[["season", "combined_competitiveness_score", "combined_competitiveness_rank"]].to_string(index=False))

    incomplete = summary[summary.get("combined_data_quality_score", pd.Series(dtype=float)) < 70]
    if "combined_data_quality_score" in summary.columns and not incomplete.empty:
        print("\nSeasons with combined_data_quality_score < 70:")
        print(incomplete[["season", "combined_data_quality_score", "data_quality_notes"]].to_string(index=False))

    for season in ("2007/08", "2008/09"):
        row = summary[summary["season"] == season]
        if not row.empty and "player_stats_assists_available" in row.columns:
            print(f"\n{season} player_stats_assists_available: {row.iloc[0]['player_stats_assists_available']}")

    print(f"\nAdvanced charts folder: {ADV_CHARTS}")


def main() -> None:
    summary, feasibility, _, advanced_created = build_season_summary()
    print_validation(summary, feasibility)
    if advanced_created:
        print_advanced_features_summary(summary, advanced_created)

    print("\n" + "=" * 60)
    print("PART 15 – FINAL CHECK")
    print("=" * 60)
    print("\n1. Created files under outputs/research_summary/:")
    for p in sorted(OUTPUT.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(ROOT)}")

    print("\n2. Title decision fix verification:")
    s0607 = summary[summary["season"] == "2006/07"].iloc[0]
    print(f"   2006/07 title_decision_round: {s0607['title_decision_round']}")
    print(f"   2006/07 title_decision_method: {s0607['title_decision_method']}")
    print(f"   2006/07 rounds_before_end_when_title_decided: {s0607['rounds_before_end_when_title_decided']}")

    audit_path = OUTPUT / "title_decision_audit.csv"
    if audit_path.exists():
        audit_df = pd.read_csv(audit_path)
        audit_0607 = audit_df[
            (audit_df["season"] == "2006/07")
            & (audit_df["global_round"].between(10, 15))
        ].sort_values("global_round")
        print("\n   Audit rows 2006/07 rounds 10–15:")
        print(audit_0607.to_string(index=False))

    fallback = summary[
        summary.get("title_decision_method", pd.Series(dtype=str)) == "regular_only_fallback"
    ]
    print("\n3. Playoff seasons with regular_only_fallback:")
    if fallback.empty:
        print("   (none)")
    else:
        print(fallback[["season", "title_decision_round", "title_decision_method"]].to_string(index=False))

    print("\n4. Relegation race fix verification:")
    s0607 = summary[summary["season"] == "2006/07"].iloc[0]
    s1718 = summary[summary["season"] == "2017/18"].iloc[0]
    print(f"   2006/07 teams_in_relegation_race_last_5_rounds: {s0607['teams_in_relegation_race_last_5_rounds']}")
    print(f"   2006/07 relegation_race_method: {s0607['relegation_race_method']}")
    print(f"   2017/18 teams_in_relegation_race_last_5_rounds: {s1718['teams_in_relegation_race_last_5_rounds']}")
    print(f"   2017/18 teams_in_relegation_race_last_5_regular_rounds: {s1718['teams_in_relegation_race_last_5_regular_rounds']}")
    print(f"   2017/18 relegation_race_method: {s1718['relegation_race_method']}")

    playoff_seasons = summary[summary["season"].apply(is_playoff_season)]
    print("\n   Playoff-era relegation_race_method:")
    print(playoff_seasons[["season", "relegation_race_method"]].to_string(index=False))

    missing_rel = []
    for _, r in playoff_seasons.iterrows():
        slug = r["season"].replace("/", "_")
        if not (DATA / "relegation_playoff_tracking" / f"relegation_playoff_tracking_{slug}.csv").exists():
            missing_rel.append(r["season"])
    print("\n   Playoff seasons missing relegation_playoff_tracking:")
    print(f"   {missing_rel if missing_rel else '(none)'}")

    print("\n5. First 10 rows – final_report_research_table.csv:")
    fr = pd.read_csv(OUTPUT / "final_report_research_table.csv")
    print(fr.head(10).to_string(index=False))

    print("\n6. First 5 rows – research_question_feasibility.csv:")
    print(feasibility.head().to_string(index=False))

    att_completed = summary[summary["is_completed_season"]]
    att_filled = att_completed[att_completed["attendance_available"].fillna(False)]
    print(f"\n7. Attendance: {len(att_filled)} seasons computed, "
          f"{len(att_completed) - len(att_filled)} left empty (coverage < {ATTENDANCE_MIN_COVERAGE_PCT:.0f}%)")

    nan_cols = []
    completed = summary[summary["is_completed_season"]]
    for col in completed.columns:
        if col.startswith("attendance_"):
            continue
        if completed[col].isna().all():
            nan_cols.append(col)
    print("\n8. Metrics all-NaN for completed seasons:")
    print(nan_cols if nan_cols else "   (none)")

    careful = summary[summary["data_quality_flag"].astype(str).str.contains("backup|incomplete|missing", case=False, na=False)]
    print("\n9. Seasons to treat carefully:")
    for _, r in careful.iterrows():
        print(f"   {r['season']}: {r['data_quality_flag']}")


if __name__ == "__main__":
    main()
