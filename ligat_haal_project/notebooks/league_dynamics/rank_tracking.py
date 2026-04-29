"""
Round-by-round league rank changes from match results (Israeli Ligat Ha'Al).

Regular season: standings rebuilt from TM match CSV rounds.
Playoffs (split era): separate streams per subgroup after halving regular carry points.
  - Seasons 2009/10 and 2010/11: חוק הקיזוז — integer halving pts // 2 (explicit in outputs).
  - Other split-era seasons: int(round(pts / 2)) — documented in notes.

Match sources:
  - Primary: data/matches/matches_<season>_transfermarkt[_dated].csv when playoffs extend past regular_last.
  - Fallback: data/playoffs/scraped_betexplorer/championship|relegation_<season>.csv

Subgroup membership (canonical names): first playoff snapshot rows from TM tracking CSVs.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import pandas as pd

EXPLICIT_TRUNC_HALF_SEASONS = frozenset({"2009/10", "2010/11"})
LEGACY_NO_PLAYOFF_SEASONS = frozenset({"2006/07", "2007/08", "2008/09"})


def find_data_root(start: Path | None = None) -> Path:
    root = start or Path(__file__).resolve()
    for _ in range(10):
        if (root / "data").is_dir():
            return root / "data"
        root = root.parent
    raise FileNotFoundError("Could not locate project data/ directory.")


DATA = find_data_root()
MATCHES_DIR = DATA / "matches"
REG_TRACK_DIR = DATA / "regular_season_tracking"
CHAMP_TRACK_DIR = DATA / "championship_playoff_tracking"
REL_TRACK_DIR = DATA / "relegation_playoff_tracking"
BETEXP_DIR = DATA / "playoffs" / "scraped_betexplorer"


def season_safe(season: str) -> str:
    return season.replace("/", "_")


def load_matches_tm(season: str) -> pd.DataFrame:
    safe = season_safe(season)
    candidates = [
        MATCHES_DIR / f"matches_{safe}_ligat_haal_transfermarkt_dated.csv",
        MATCHES_DIR / f"matches_{safe}_ligat_haal_transfermarkt.csv",
    ]
    picked: Optional[Path] = None
    for p in candidates:
        if p.is_file():
            picked = p
            break
    if picked is None:
        raise FileNotFoundError(f"No TM match file for {season} under {MATCHES_DIR}")
    df = pd.read_csv(picked)
    need = {"home", "away", "score", "round"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{picked.name} missing columns {miss}")
    df = df.dropna(subset=["home", "away", "score"]).copy()
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)
    return df


def _read_tracking_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_tracking_bundle(season: str) -> pd.DataFrame:
    safe = season_safe(season)
    parts = [
        _read_tracking_file(REG_TRACK_DIR / f"regular_season_tracking_{safe}.csv"),
        _read_tracking_file(CHAMP_TRACK_DIR / f"championship_playoff_tracking_{safe}.csv"),
        _read_tracking_file(REL_TRACK_DIR / f"relegation_playoff_tracking_{safe}.csv"),
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[df["season"].astype(str) == season]
    return df


def regular_last_round(track: pd.DataFrame) -> Optional[int]:
    if track.empty:
        return None
    reg = track[track["stage"].astype(str).str.strip().str.lower().eq("regular")]
    if reg.empty:
        return None
    return int(pd.to_numeric(reg["round"], errors="coerce").max())


def pool_at_stage_first_round(
    track: pd.DataFrame, stage: str
) -> Optional[tuple[int, frozenset[str]]]:
    st = track[track["stage"].astype(str).str.strip().str.lower() == stage]
    if st.empty:
        return None
    mn = int(pd.to_numeric(st["round"], errors="coerce").min())
    row0 = st[pd.to_numeric(st["round"], errors="coerce") == mn]
    teams = frozenset(row0["team"].astype(str).str.strip().unique())
    return mn, teams


def parse_score(s) -> tuple[int, int]:
    p = str(s).strip().replace(" ", "").split(":")
    return int(p[0]), int(p[1])


@dataclass
class TeamStat:
    pts: int = 0
    gf: int = 0
    ga: int = 0


def gd_of(s: TeamStat) -> int:
    return s.gf - s.ga


def apply_result(st: dict[str, TeamStat], home: str, away: str, hg: int, ag: int) -> None:
    sh, sa = st[home], st[away]
    sh.gf += hg
    sh.ga += ag
    sa.gf += ag
    sa.ga += hg
    if hg > ag:
        sh.pts += 3
    elif hg < ag:
        sa.pts += 3
    else:
        sh.pts += 1
        sa.pts += 1


def table_from_stats(
    st: dict[str, TeamStat], teams: Iterable[str]
) -> list[tuple[str, int, int, int, int]]:
    rows = []
    for t in teams:
        s = st[t]
        rows.append((t, s.pts, gd_of(s), s.gf))
    rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    return rows


def assign_ranks(sorted_rows: list[tuple[str, int, int, int, int]]) -> dict[str, int]:
    return {t: i + 1 for i, (t, *_rest) in enumerate(sorted_rows)}


def validate_ranks(ranks: dict[str, int], n_teams: int) -> list[str]:
    errs: list[str] = []
    vs = sorted(ranks.values())
    if len(vs) != n_teams:
        errs.append(f"rank count {len(vs)} != n_teams {n_teams}")
    if len(set(vs)) != len(vs):
        errs.append("duplicate rank numbers in table")
    if len(vs) == n_teams and set(vs) != set(range(1, n_teams + 1)):
        errs.append(f"rank values not 1..{n_teams}: {sorted(set(vs))}")
    return errs


def half_regular_points(pts: int, season: str) -> int:
    if season in EXPLICIT_TRUNC_HALF_SEASONS:
        return pts // 2
    return int(round(pts / 2))


def simulate_regular_through_round(
    matches: pd.DataFrame, team_universe: set[str], max_r: int
) -> dict[str, TeamStat]:
    st = {t: TeamStat() for t in team_universe}
    sub = matches[matches["round"] <= max_r]
    for _, r in sub.iterrows():
        h, a = r["home"], r["away"]
        if h not in st or a not in st:
            continue
        hg, ag = parse_score(r["score"])
        apply_result(st, h, a, hg, ag)
    return st


def diff_ranks(
    prev: dict[str, int], new: dict[str, int], teams: Iterable[str]
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for t in teams:
        if t not in prev or t not in new:
            continue
        if prev[t] != new[t]:
            out.append((t, prev[t], new[t]))
    out.sort(key=lambda x: x[0])
    return out


def _norm_key(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", x.lower().replace("'", ""))


def map_name_to_pool(name: str, pool: frozenset[str]) -> str:
    if name in pool:
        return name
    nk = _norm_key(name)
    for c in pool:
        if _norm_key(c) == nk:
            return c
    for c in pool:
        if nk in _norm_key(c) or _norm_key(c) in nk:
            return c
    return name


def load_betexplorer_playoff(
    season: str, kind: Literal["championship", "relegation"]
) -> Optional[pd.DataFrame]:
    safe = season_safe(season)
    p = BETEXP_DIR / f"{kind}_{safe}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    for c in ("home_team", "away_team", "home_goals", "away_goals", "round"):
        if c not in df.columns:
            raise ValueError(f"{p.name} missing {c}")
    return df


def betexplorer_to_events(
    df: pd.DataFrame, map_name: Callable[[str], str]
) -> list[tuple[int, str, str, int, int]]:
    rows: list[tuple[int, str, str, int, int]] = []
    for rnd in sorted(df["round"].unique()):
        sub = df[df["round"] == rnd]
        for _, r in sub.iterrows():
            h = map_name(str(r["home_team"]).strip())
            a = map_name(str(r["away_team"]).strip())
            rows.append(
                (
                    int(rnd),
                    h,
                    a,
                    int(r["home_goals"]),
                    int(r["away_goals"]),
                )
            )
    return rows


def tm_playoff_slice(
    matches: pd.DataFrame,
    reg_last: int,
    pool: frozenset[str],
    map_name: Callable[[str], str],
) -> list[tuple[int, str, str, int, int]]:
    sub = matches[matches["round"] > reg_last].copy()
    ev: list[tuple[int, str, str, int, int]] = []
    for _, r in sub.sort_values(["round"]).iterrows():
        h = map_name(str(r["home"]).strip())
        a = map_name(str(r["away"]).strip())
        if h not in pool or a not in pool:
            continue
        hg, ag = parse_score(r["score"])
        ev.append((int(r["round"]), h, a, hg, ag))
    return ev


def group_events_by_tm_round(
    events: list[tuple[int, str, str, int, int]]
) -> dict[int, list[tuple[str, str, int, int]]]:
    by: dict[int, list[tuple[str, str, int, int]]] = defaultdict(list)
    for rn, h, a, hg, ag in events:
        by[rn].append((h, a, hg, ag))
    return dict(by)


def group_events_by_subround(
    events: list[tuple[int, str, str, int, int]]
) -> dict[int, list[tuple[str, str, int, int]]]:
    return group_events_by_tm_round(events)


@dataclass
class SeasonNotes:
    season: str
    points_halved_before_playoffs: bool
    explicit_kizuz_rule: bool
    regular_last_round: Optional[int]
    playoff_source: str
    messages: list[str] = field(default_factory=list)


def season_from_filename_regular(name: str) -> Optional[str]:
    m = re.search(r"regular_season_tracking_(\d{4})_(\d{2})\.csv$", name)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


@dataclass
class RankChangeRow:
    season: str
    stage: str
    segment_label: str
    team: str
    prev_rank: int
    new_rank: int


@dataclass
class RankChangeBlock:
    season: str
    stage: str
    label: str
    changes: list[tuple[str, int, int]]


def blocks_to_rows(blocks: Iterable[RankChangeBlock]) -> list[RankChangeRow]:
    rows: list[RankChangeRow] = []
    for b in blocks:
        for t, a, c in b.changes:
            rows.append(RankChangeRow(b.season, b.stage, b.label, t, a, c))
    return rows


def sanity_jump_warning(
    changes: list[tuple[str, int, int]], n_teams: int
) -> Optional[str]:
    mx = 0
    for _t, a, b in changes:
        mx = max(mx, abs(a - b))
    if mx > n_teams - 1:
        return f"largest rank jump ({mx}) exceeds theoretical max {n_teams - 1}"
    return None


def playoff_initial_stats(
    pool: frozenset[str],
    st_reg_fin: dict[str, TeamStat],
    season: str,
    validation: list[str],
    reg_label: str,
) -> dict[str, TeamStat]:
    init: dict[str, TeamStat] = {}
    for t in pool:
        base = st_reg_fin.get(t)
        if base is None:
            validation.append(f"{reg_label}: team {t} missing from regular simulation")
            continue
        hp = half_regular_points(base.pts, season)
        init[t] = TeamStat(pts=hp, gf=base.gf, ga=base.ga)
    return init


def analyze_season(season: str) -> tuple[SeasonNotes, list[RankChangeBlock], list[str]]:
    validation: list[str] = []
    blocks: list[RankChangeBlock] = []
    track = load_tracking_bundle(season)
    reg_last = regular_last_round(track)

    notes = SeasonNotes(
        season=season,
        points_halved_before_playoffs=False,
        explicit_kizuz_rule=season in EXPLICIT_TRUNC_HALF_SEASONS,
        regular_last_round=reg_last,
        playoff_source="none",
        messages=[],
    )

    matches = load_matches_tm(season)
    all_teams = set(matches["home"]) | set(matches["away"])
    max_tm_r = int(matches["round"].max())

    if reg_last is None:
        validation.append(f"{season}: no regular rows in tracking; using max TM round.")
        reg_last = max_tm_r

    # --- Regular ---
    prev_ranks: Optional[dict[str, int]] = None
    for rnd in range(1, reg_last + 1):
        st = simulate_regular_through_round(matches, all_teams, rnd)
        tab = table_from_stats(st, sorted(st.keys()))
        ranks = assign_ranks(tab)
        for e in validate_ranks(ranks, len(st)):
            validation.append(f"{season} regular r{rnd}: {e}")

        if prev_ranks is not None:
            chg = diff_ranks(prev_ranks, ranks, st.keys())
            w = sanity_jump_warning(chg, len(st))
            if w:
                validation.append(f"{season} regular r{rnd}: {w}")
            label = f"Regular: after round {rnd} vs after round {rnd - 1}"
            blocks.append(RankChangeBlock(season, "regular", label, chg))

        prev_ranks = ranks

    n_reg_teams = len({*matches["home"], *matches["away"]})
    if n_reg_teams != len(prev_ranks or {}):
        validation.append(f"{season}: regular team count inconsistent in last round")

    if season in LEGACY_NO_PLAYOFF_SEASONS:
        notes.messages.append("Legacy era: no playoff split tracked for this project.")
        return notes, blocks, validation

    ch = pool_at_stage_first_round(track, "championship_playoff")
    rel = pool_at_stage_first_round(track, "relegation_playoff")
    if ch is None and rel is None:
        notes.messages.append("No championship/relegation tracking — playoffs skipped.")
        return notes, blocks, validation

    champ_pool = ch[1] if ch else frozenset()
    rel_pool = rel[1] if rel else frozenset()

    st_reg_fin = simulate_regular_through_round(matches, all_teams, reg_last)
    notes.points_halved_before_playoffs = True
    notes.messages.append(
        "Subgroup tables after the split: ranks are within the championship or relegation cohort only "
        "(not comparable to pre-split global rank 1–N)."
    )
    notes.messages.append(
        "Points from the regular season are halved for playoff carry points; goal difference from the "
        "regular phase is carried for tie-breaks within each mini-league."
    )
    if notes.explicit_kizuz_rule:
        notes.messages.append(
            "חוק הקיזוז seasons (2009/10, 2010/11): carry points halved with integer truncation (pts // 2)."
        )
    else:
        notes.messages.append(
            "Other split-era seasons: carry points mapped with int(round(pts / 2)) before playoff fixtures."
        )

    tm_extends = max_tm_r > reg_last

    for stage_name, pool, tag in (
        ("championship", champ_pool, "championship_playoff"),
        ("relegation", rel_pool, "relegation_playoff"),
    ):
        if not pool:
            continue
        map_n = lambda x, p=pool: map_name_to_pool(x, p)  # type: ignore[misc]

        ev: list[tuple[int, str, str, int, int]] = []
        if tm_extends:
            ev = tm_playoff_slice(matches, reg_last, pool, map_n)
            if ev:
                notes.playoff_source = "transfermarkt"

        if not ev:
            bdf = load_betexplorer_playoff(season, stage_name)  # type: ignore[arg-type]
            if bdf is not None:
                ev = betexplorer_to_events(bdf, map_n)
                notes.playoff_source = "betexplorer"

        if not ev:
            validation.append(f"{season}: missing {stage_name} playoff fixtures.")
            continue

        init_st = playoff_initial_stats(pool, st_reg_fin, season, validation, season)
        sub_work = {t: TeamStat(**init_st[t].__dict__) for t in init_st}
        ranks_carry_only = assign_ranks(
            table_from_stats(sub_work, sorted(sub_work.keys()))
        )
        by_sub = group_events_by_subround(ev)

        sorted_subs = sorted(by_sub.keys())
        if not sorted_subs:
            validation.append(f"{season}: {stage_name} playoff events grouped empty.")
            continue

        prev = ranks_carry_only
        for idx, subr in enumerate(sorted_subs):
            for h, a, hg, ag in by_sub[subr]:
                if h in sub_work and a in sub_work:
                    apply_result(sub_work, h, a, hg, ag)
            cur = assign_ranks(table_from_stats(sub_work, sorted(sub_work.keys())))
            for e in validate_ranks(cur, len(sub_work)):
                validation.append(f"{season} {stage_name} mini-r{subr}: {e}")
            chg = diff_ranks(prev, cur, sub_work.keys())
            w = sanity_jump_warning(chg, len(sub_work))
            if w:
                validation.append(f"{season} {stage_name} mini-r{subr}: {w}")
            prev_label = (
                "halved-carry table (before any playoff match)"
                if idx == 0
                else f"after mini-league round {sorted_subs[idx - 1]}"
            )
            label = (
                f"{stage_name} playoff: after mini-league round {subr} vs {prev_label}"
            )
            blocks.append(RankChangeBlock(season, tag, label, chg))
            prev = cur

    return notes, blocks, validation


def format_season_report(
    notes: SeasonNotes, blocks: list[RankChangeBlock], validations: list[str]
) -> str:
    lines: list[str] = []
    s = notes.season
    lines.append(f"Season {s}:")
    lines.append("  Notes:")
    lines.append(f"    - Points halved before playoff groups: {notes.points_halved_before_playoffs}")
    if notes.explicit_kizuz_rule:
        lines.append(
            "    - SPECIAL (2009/10 or 2010/11): חוק הקיזוז — carry points truncated with integer division pts // 2"
        )
    else:
        lines.append(
            "    - Carry points before playoffs: int(round(pts / 2)) unless legacy exception above applies"
        )
    lines.append(f"    - Regular season ended after TM round {notes.regular_last_round}")
    lines.append(f"    - Playoff fixture source where applicable: {notes.playoff_source}")
    for m in notes.messages:
        lines.append(f"    - {m}")
    if validations:
        lines.append("  Validation warnings:")
        for v in validations:
            lines.append(f"    ! {v}")

    if not blocks:
        lines.append("  (No rank-change blocks produced.)")
        return "\n".join(lines)

    for blk in blocks:
        lines.append(f"  [{blk.stage}] {blk.label}")
        if blk.changes:
            for team, a, b in blk.changes:
                lines.append(f"    {team}: rank {a} -> {b}")
        else:
            lines.append("    No ranking changes in this round")
    return "\n".join(lines)


def rows_to_dataframe(rows: list[RankChangeRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "stage",
                "segment_label",
                "team",
                "prev_rank",
                "new_rank",
            ]
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def discover_seasons() -> list[str]:
    seasons: set[str] = set()
    for p in REG_TRACK_DIR.glob("regular_season_tracking_*.csv"):
        lab = season_from_filename_regular(p.name)
        if lab:
            seasons.add(lab)
    return sorted(seasons)


def run_all_seasons(
    seasons: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seasons is None:
        seasons = discover_seasons()
    rows_acc: list[RankChangeRow] = []
    note_payload: list[dict] = []
    for sea in seasons:
        notes, blocks, val = analyze_season(sea)
        note_payload.append(
            {
                "season": notes.season,
                "explicit_kizuz": notes.explicit_kizuz_rule,
                "regular_last_round": notes.regular_last_round,
                "playoff_source": notes.playoff_source,
                "warnings": json.dumps(val, ensure_ascii=False),
                "notes": " | ".join(notes.messages),
            }
        )
        rows_acc.extend(blocks_to_rows(blocks))
    return rows_to_dataframe(rows_acc), pd.DataFrame(note_payload)


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="Ligat HaAl rank dynamics from matches")
    pa.add_argument("--season", nargs="*", help="e.g. 2015/16 (default: all)")
    pa.add_argument("--out", type=Path, help="CSV output dir")
    ns = pa.parse_args()
    seasons_list = ns.season if ns.season else discover_seasons()
    rdfs: list[pd.DataFrame] = []
    txt_parts: list[str] = []
    for sea in seasons_list:
        n, blk, va = analyze_season(sea)
        txt_parts.append(format_season_report(n, blk, va))
        txt_parts.append("")
        rdfs.append(rows_to_dataframe(blocks_to_rows(blk)))

    merged = pd.concat(rdfs, ignore_index=True) if rdfs else rows_to_dataframe([])
    outp = Path(__file__).resolve().parent.parent / "data" / "processed" / "rank_dynamics"
    out_dir = Path(ns.out) if ns.out else outp
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "rank_changes_all.csv", index=False, encoding="utf-8-sig")
    (out_dir / "rank_changes_report.txt").write_text("\n".join(txt_parts), encoding="utf-8")
    report_tail = format_season_report(n, blk, va)[:3000]
    print(report_tail.encode("ascii", errors="replace").decode("ascii"))
