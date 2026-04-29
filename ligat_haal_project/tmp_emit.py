import json
from pathlib import Path

ROOT = Path("notebooks")

def md(s):
    lines = [ln + "\n" for ln in s.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(s):
    lines = [ln + "\n" for ln in s.strip().split("\n")]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}

cells = []

cells.append(md("""
# Relegation battle competitiveness — Israeli Premier League

This notebook computes four relegation-focused competitiveness metrics from **existing project files only**.

**Outputs** are saved under:

`notebooks/data/processed/relegation_competitiveness/`

No original source CSVs are modified.
"""))

cells.append(md("""
## Missing Data Report

The following capabilities depend on repository assets:

| Asset | Needed for | Notes |
|-------|------------|------|
| `data/matches/matches_*_ligat_haal_transfermarkt_dated.csv` (preferred) or `*_transfermarkt.csv` | Metrics 1, 2, incremental checkpoints | Needed to reconstruct **points** standings after each **regular-season** round. Seasons without a match file cannot get those metrics. |
| `data/processed/position_tables_by_round_tm/positions_by_round_*_tm.csv` | Metric 4 (`rank`-based volatility) | Team columns must align with Transfermarkt short names used in matches. Missing seasons skipped for metric 4. |

**Important limitations**

1. **Regular season vs playoffs:** Incremental standings are built **only from regular-season fixtures** (double round-robin portion). After the league splits, relegation outcomes also depend on the relegation playoff; metrics here describe **regular-season relegation-zone dynamics**, not post-split totals.
2. **Metric 2 (mathematical elimination):** We use a conservative ordering by **maximum possible remaining points** (each team assumes all remaining wins). Tie-break ambiguity at equal max points is noted in output. Where matches are incomplete, elimination rounds are unreliable.

If any column is absent (e.g. missing `score` split), the affected season rows are flagged or omitted.
"""))

cells.append(md("## Section 0 — Paths, imports, and helper functions"))

cells.append(code(r"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

# --- Resolve notebooks root (contains data/)
NOTEBOOK_ROOT = Path(".").resolve()
for _ in range(6):
    if (NOTEBOOK_ROOT / "data").is_dir():
        break
    NOTEBOOK_ROOT = NOTEBOOK_ROOT.parent
else:
    NOTEBOOK_ROOT = Path(".").resolve()

DATA = NOTEBOOK_ROOT / "data"
INTERIM = DATA / "interim"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MATCHES_DIR = DATA / "matches"
REG_FINAL_DIR = INTERIM / "scraped_standings" / "regular_final_tables"
POSITION_TM_DIR = PROCESSED / "position_tables_by_round_tm"
RELEG_COLLECTION = INTERIM / "relegation_struggles"

OUT_DIR = PROCESSED / "relegation_competitiveness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Notebook data root: {DATA}")
print(f"Output folder: {OUT_DIR}")
"""))

cells.append(code(r"""
def season_label_from_regular_filename(name: str) -> Optional[str]:
    """Parse 'regular_final_table_YYYY_YY.csv' → 'YYYY/YY'."""
    m = re.match(r"regular_final_table_(\d{4})_(\d{2})\.csv$", name, re.I)
    if not m:
        return None
    return f"{int(m.group(1))}/{m.group(2)}"


def relegation_slots(league_size: int) -> int:
    """Our project assumes 2 relegated clubs for 12-, 14-, and 16-team eras."""
    if league_size not in (12, 14, 16):
        raise ValueError(league_size)
    return 2


def find_matches_file(season: str) -> Optional[Path]:
    """Prefer dated Transfermarkt aggregates (round + score)."""
    safe = season.replace("/", "_")
    candidates = [
        MATCHES_DIR / f"matches_{safe}_ligat_haal_transfermarkt_dated.csv",
        MATCHES_DIR / f"matches_{safe}_ligat_haal_transfermarkt.csv",
        RAW / f"matches_{safe}_ligat_haal_transfermarkt.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def find_position_tm_file(season: str) -> Optional[Path]:
    safe = season.replace("/", "_")
    p = POSITION_TM_DIR / f"positions_by_round_{safe}_tm.csv"
    return p if p.is_file() else None


def parse_score(score: str) -> tuple[int, int]:
    parts = str(score).replace(" ", "").split(":")
    return int(parts[0]), int(parts[1])


def load_regular_final_table(season: str) -> Optional[pd.DataFrame]:
    safe = season.replace("/", "_")
    p = REG_FINAL_DIR / f"regular_final_table_{safe}.csv"
    return pd.read_csv(p) if p.is_file() else None
"""))

cells.append(code(r"""
def standings_up_to_round(matches: pd.DataFrame, max_round: int) -> pd.DataFrame:
    """Rebuild W-D-L / Pts / GF / GA from match rows with round <= max_round."""
    df = matches[matches["round"] <= max_round].copy()
    teams: set[str] = set(df["home"]).union(set(df["away"]))
    z = {t: {"played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0} for t in teams}
    for _, r in df.iterrows():
        h, a = r["home"], r["away"]
        hg, ag = parse_score(r["score"])
        z[h]["gf"] += hg
        z[h]["ga"] += ag
        z[a]["gf"] += ag
        z[a]["ga"] += hg
        z[h]["played"] += 1
        z[a]["played"] += 1
        if hg > ag:
            z[h]["pts"] += 3
            z[h]["w"] += 1
            z[a]["l"] += 1
        elif hg < ag:
            z[a]["pts"] += 3
            z[a]["w"] += 1
            z[h]["l"] += 1
        else:
            z[h]["pts"] += 1
            z[a]["pts"] += 1
            z[h]["d"] += 1
            z[a]["d"] += 1
    rows = []
    for t, s in z.items():
        gd = s["gf"] - s["ga"]
        rows.append(
            {
                "team": t,
                "played": s["played"],
                "win": s["w"],
                "draw": s["d"],
                "loss": s["l"],
                "gf": s["gf"],
                "ga": s["ga"],
                "goal_diff": gd,
                "points": s["pts"],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["points", "goal_diff", "gf"], ascending=[False, False, False]
    ).reset_index(drop=True)


def rounds_played_per_team(matches_full: pd.DataFrame) -> pd.Series:
    """Count scheduled regular matches per team up to max round in file."""
    rmax = int(matches_full["round"].max())
    teams = set(matches_full["home"]).union(set(matches_full["away"]))
    cnt = {t: 0 for t in teams}
    for _, r in matches_full.iterrows():
        cnt[r["home"]] += 1
        cnt[r["away"]] += 1
    return pd.Series(cnt)


def max_possible elimination TODO
"""))

