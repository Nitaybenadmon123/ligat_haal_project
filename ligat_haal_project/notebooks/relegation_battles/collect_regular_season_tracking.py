#!/usr/bin/env python3
"""
Round-by-round Ligat ha'Al REGULAR season standings from Transfermarkt.

Same scraping contract as collect_relegation_playoff_tracking.py and
scrape_transfermarkt_championship_playoff_full_season in
notebooks/title_race/01_title_race_collection.ipynb:
  table.items tbody tr, cells[0] rank, [2] team, [8] goal_diff, [9] points.

League: ISR1 (full season, not split phase).
saison_id = int(season.split(\"/\")[0]).

Rows are ordered by TM; round = Transfermarkt Spieltag for the regular fixture list.
Expected number of matchdays is taken from the project's match file (max(round)).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
DATA = NOTEBOOK_ROOT / "data"
MATCHES_DIR = DATA / "matches"
OUT_DIR = DATA / "regular_season_tracking"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
}

TM_REGULAR_SPIELTAG = (
    "https://www.transfermarkt.com/ligat-haal/spieltagtabelle/wettbewerb/ISR1"
)

# Safety cap (12-team era = 33 RR; buffer)
MAX_SPIELTAG_HARD = 40

REQUEST_PAUSE_SEC = 0.12

COLS_OUT = [
    "season",
    "stage",
    "round",
    "team",
    "points",
    "position",
    "goal_diff",
]


def _safe_int(x) -> int:
    s = str(x).strip().replace("\u2212", "-").replace(",", "")
    if not s:
        raise ValueError("empty int field")
    return int(s)


def match_file(season_slash: str) -> Path | None:
    safe = season_slash.replace("/", "_")
    for name in (
        f"matches_{safe}_ligat_haal_transfermarkt_dated.csv",
        f"matches_{safe}_ligat_haal_transfermarkt.csv",
    ):
        p = MATCHES_DIR / name
        if p.is_file():
            return p
    return None


def expected_regular_rounds(season_slash: str) -> int | None:
    p = match_file(season_slash)
    if p is None:
        return None
    df = pd.read_csv(p, usecols=["round"])
    return int(df["round"].max())


def discover_seasons_from_matches() -> list[str]:
    """Season labels from per-season match files (dated preferred; then undated)."""
    seen: set[str] = set()
    order: list[str] = []

    def add_from_name(fname: str) -> None:
        m = re.match(
            r"^matches_(\d{4})_(\d{2})_ligat_haal_transfermarkt(?:_dated)?\.csv$",
            fname,
        )
        if not m:
            return
        y1, y2 = int(m.group(1)), int(m.group(2))
        label = f"{y1}/{y2:02d}"
        if label not in seen:
            seen.add(label)
            order.append(label)

    for p in sorted(MATCHES_DIR.glob("matches_*_*_ligat_haal_transfermarkt_dated.csv")):
        add_from_name(p.name)
    for p in sorted(MATCHES_DIR.glob("matches_*_*_ligat_haal_transfermarkt.csv")):
        add_from_name(p.name)
    return order


def scrape_transfermarkt_regular_season_full_season(
    season_str: str, saison_id: int, regular_rounds_cap: int
) -> pd.DataFrame | None:
    """
    Scrape ISR1 standings after each regular Spieltag (points as shown on TM).
    Stops after regular_rounds_cap Spieltage, or on scrape failure / duplicate snapshot.
    """
    cap = min(regular_rounds_cap, MAX_SPIELTAG_HARD)
    all_dfs: list[pd.DataFrame] = []
    seen: set[str] = set()
    n_teams_expect: int | None = None

    for spieltag in range(1, cap + 1):
        url = f"{TM_REGULAR_SPIELTAG}?saison_id={saison_id}&spieltag={spieltag}"
        r = requests.get(url, headers=HEADERS, timeout=25)
        time.sleep(REQUEST_PAUSE_SEC)
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", class_="items")
        if table is None or table.find("tbody") is None:
            break

        standings: list[dict] = []
        for row in table.find("tbody").find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 10:
                continue
            try:
                standings.append(
                    {
                        "season": season_str,
                        "tm_spieltag": spieltag,
                        "rank": _safe_int(cells[0].text),
                        "team": cells[2].text.strip(),
                        "goal_diff": _safe_int(cells[8].text),
                        "points": _safe_int(cells[9].text),
                    }
                )
            except (ValueError, TypeError):
                continue

        n = len(standings)
        if not n:
            break
        if n_teams_expect is None:
            if n not in (12, 14, 16):
                raise ValueError(
                    f"{season_str} spieltag {spieltag}: unexpected league size {n} (expect 12, 14, or 16)"
                )
            n_teams_expect = n
        elif n != n_teams_expect:
            raise ValueError(
                f"{season_str} spieltag {spieltag}: row count {n} "
                f"!= first matchday ({n_teams_expect})"
            )

        teams = [s["team"] for s in standings]
        if len(teams) != len(set(teams)):
            raise ValueError(
                f'{season_str} spieltag {spieltag}: duplicate team rows in TM table'
            )

        sig = hashlib.md5(
            str([(s["team"], s["points"]) for s in standings]).encode()
        ).hexdigest()
        if sig in seen:
            break
        seen.add(sig)
        all_dfs.append(pd.DataFrame(standings))

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def validate_regular(
    raw: pd.DataFrame, season: str, n_teams: int, expected_rounds: int
) -> list[str]:
    warns: list[str] = []
    spieltags = sorted(raw["tm_spieltag"].unique())
    mx = int(max(spieltags))
    wanted = list(range(1, mx + 1))
    if spieltags != wanted:
        warns.append(
            f"{season}: missing Spieltage in scrape (have {spieltags}, wanted contiguous 1..{mx})"
        )
    if mx < expected_rounds:
        warns.append(
            f"{season}: TM returned {mx} Spieltage; match file expects {expected_rounds} rounds"
        )
    elif mx > expected_rounds:
        warns.append(
            f"{season}: TM returned {mx} Spieltage; match file max round is {expected_rounds} (check split / fixtures)"
        )

    for st in spieltags:
        sub = raw[raw["tm_spieltag"] == st]
        if len(sub) != n_teams:
            warns.append(f"{season} spieltag {st}: rows {len(sub)} != {n_teams}")

    for tm, grp in raw.groupby("team"):
        grp = grp.sort_values("tm_spieltag")
        pts = grp["points"].tolist()
        for i in range(1, len(pts)):
            if pts[i] < pts[i - 1]:
                warns.append(
                    f'{season}: team "{tm}" pts decreased ({pts[i - 1]} -> {pts[i]})'
                )
                break
    return warns


def backup_if_exists(path: Path) -> None:
    if not path.is_file():
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bak = path.with_name(path.stem + f"_bak_{ts}" + path.suffix)
    bak.write_bytes(path.read_bytes())


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "season": row["season"],
                "stage": "regular",
                "round": int(row["tm_spieltag"]),
                "team": row["team"],
                "points": row["points"],
                "position": row["rank"],
                "goal_diff": row["goal_diff"],
            }
        )
    return pd.DataFrame(rows)


def collect_one_season(season_slash: str, dry_run: bool) -> tuple[pd.DataFrame | None, list[str]]:
    log: list[str] = []
    exp_r = expected_regular_rounds(season_slash)
    if exp_r is None:
        log.append(f"WARN {season_slash}: no matches file; skip")
        return None, log

    saison_id = int(season_slash.split("/")[0])

    raw = scrape_transfermarkt_regular_season_full_season(
        season_slash, saison_id, exp_r
    )
    if raw is None or raw.empty:
        log.append(f"WARN {season_slash}: could not scrape regular ISR1 tables")
        return None, log

    if int(raw["tm_spieltag"].max()) > exp_r:
        raw = raw[raw["tm_spieltag"] <= exp_r].copy()

    n_teams = int(
        raw[raw["tm_spieltag"] == raw["tm_spieltag"].min()]["team"].nunique()
    )
    log.extend(validate_regular(raw, season_slash, n_teams, exp_r))

    df_out = build_output(raw).sort_values(["round", "position", "team"])

    print("=" * 72)
    print(f"{season_slash}: regular season | teams = {n_teams} | match file max round = {exp_r}")
    first_r = df_out["round"].min()
    last_r = df_out["round"].max()
    fp = COLS_OUT
    print("\nFirst scraped round table:")
    print(df_out[df_out["round"] == first_r][fp].to_string(index=False))
    print("\nLast scraped round table:")
    print(df_out[df_out["round"] == last_r][fp].sort_values(["position", "team"]).to_string(index=False))
    print(f"\nTM Spieltage scraped: {raw['tm_spieltag'].nunique()} (min={first_r}, max={last_r})")
    if log:
        print("Validation notes:")
        for w in log:
            print(" ", w)

    if df_out.groupby(["season", "round", "team"]).size().max() > 1:
        raise ValueError(f"{season_slash}: duplicate row season/round/team")

    if dry_run:
        return df_out, log

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file = OUT_DIR / f'regular_season_tracking_{season_slash.replace("/", "_")}.csv'
    backup_if_exists(per_file)
    df_out.to_csv(per_file, index=False, encoding="utf-8-sig")
    print(f"Saved {per_file}")
    return df_out, log


def rebuild_combined_from_per_season_files() -> Path:
    """Concatenate all regular_season_tracking_YYYY_YY.csv into one CSV."""
    paths = sorted(OUT_DIR.glob("regular_season_tracking_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No regular_season_tracking_*.csv in {OUT_DIR}")
    dfs = []
    for p in paths:
        if p.name.startswith("all_seasons_regular"):
            continue
        dfs.append(pd.read_csv(p))
    out = pd.concat(dfs, ignore_index=True).sort_values(
        ["season", "round", "position", "team"]
    )
    combined_path = OUT_DIR / "all_seasons_regular_tracking.csv"
    backup_if_exists(combined_path)
    out.to_csv(combined_path, index=False, encoding="utf-8-sig")
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--season", help='one season label e.g. "2024/25"')
    parser.add_argument(
        "--rebuild-combined-only",
        action="store_true",
        help="Only merge existing per-season CSVs into all_seasons_regular_tracking.csv",
    )
    args = parser.parse_args()

    if args.rebuild_combined_only:
        p = rebuild_combined_from_per_season_files()
        print(f"Merged: {p} ({pd.read_csv(p).shape[0]} rows)")
        return

    seasons = [args.season] if args.season else discover_seasons_from_matches()
    if not seasons:
        print("No seasons found (missing matches_*_*_transfermarkt_dated.csv)")
        return

    all_parts: list[pd.DataFrame] = []
    for s in seasons:
        try:
            chunk, _ = collect_one_season(s, dry_run=args.dry_run)
            if chunk is not None:
                all_parts.append(chunk)
        except Exception as e:
            print(f"ERROR {s}: {e}")
            raise

    if all_parts and not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        combined_path = OUT_DIR / "all_seasons_regular_tracking.csv"
        backup_if_exists(combined_path)
        pd.concat(all_parts, ignore_index=True).sort_values(
            ["season", "round", "position", "team"]
        ).to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\nCombined: {combined_path}")


if __name__ == "__main__":
    main()
