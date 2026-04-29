#!/usr/bin/env python3
"""
Build round-by-round relegation-playoff standings from Transfermarkt.

Mirrors scrape_transfermarkt_championship_playoff_full_season in
notebooks/title_race/01_title_race_collection.ipynb (same table markup,
requests + BeautifulSoup, table.items, cell indices 0 rank / 2 team / … / 9 pts).

Competition: Ligat ha'Al – Relegation round (ISRA), Spieltag tables.
Season id on TM: saison_id = int(season.split('/')[0]) (e.g. 2024/25 -> 2024).

Continuous round numbering: round = max_regular_round + relegation_playoff_matchday (TM spieltag).
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
DATA = NOTEBOOK_ROOT / "data"
MATCHES_DIR = DATA / "matches"
OUT_DIR = DATA / "relegation_playoff_tracking"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
}

TM_RELEG_PLAYOFF_SPIELTAG = (
    "https://www.transfermarkt.com/"
    "ligat-haal-relegation-round/spieltagtabelle/wettbewerb/ISRA"
)

SEASONS = [
    "2009/10",
    "2010/11",
    "2011/12",
    "2012/13",
    "2013/14",
    "2014/15",
    "2015/16",
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
    "2023/24",
    "2024/25",
]

MAX_PLAYOFF_ROUNDS = 14

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


def final_regular_round(season_slash: str) -> float | None:
    p = match_file(season_slash)
    if p is None:
        return None
    df = pd.read_csv(p, usecols=["round"])
    return float(df["round"].max())


def scrape_transfermarkt_relegation_playoff_full_season(
    season_str: str, saison_id: int
) -> pd.DataFrame | None:
    """Same pattern as championship playoff scraper; 6-8 teams per ISRA table."""
    all_dfs: list[pd.DataFrame] = []
    seen: set[str] = set()

    for spieltag in range(1, MAX_PLAYOFF_ROUNDS + 1):
        url = f"{TM_RELEG_PLAYOFF_SPIELTAG}?saison_id={saison_id}&spieltag={spieltag}"
        r = requests.get(url, headers=HEADERS, timeout=25)
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
                        "tm_playoff_spieltag": spieltag,
                        "rank": _safe_int(cells[0].text),
                        "team": cells[2].text.strip(),
                        "goal_diff": _safe_int(cells[8].text),
                        "points": _safe_int(cells[9].text),
                    }
                )
            except (ValueError, TypeError):
                continue

        n = len(standings)
        if not (6 <= n <= 8):
            break

        teams = [s["team"] for s in standings]
        if len(teams) != len(set(teams)):
            raise ValueError(
                f'{season_str} spieltag {spieltag}: duplicate team rows in table'
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


def validate_tables(raw: pd.DataFrame, season: str, n_teams_playoff: int) -> list[str]:
    warns: list[str] = []
    if not (6 <= n_teams_playoff <= 8):
        warns.append(f"{season}: playoff team count {n_teams_playoff} outside 6-8 (check TM)")

    spieltags = sorted(raw["tm_playoff_spieltag"].unique())
    wanted = list(range(1, int(max(spieltags)) + 1))
    if spieltags != wanted:
        warns.append(f"{season}: missing matchdays (have {spieltags}, wanted {wanted})")

    for st in spieltags:
        sub = raw[raw["tm_playoff_spieltag"] == st]
        if len(sub) != n_teams_playoff:
            warns.append(f"{season} spieltag {st}: row count {len(sub)} != {n_teams_playoff}")

    for tm, grp in raw.groupby("team"):
        grp = grp.sort_values("tm_playoff_spieltag")
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


def build_output_for_season(
    season_slash: str, raw: pd.DataFrame, final_r: float
) -> pd.DataFrame:
    out = []
    for _, row in raw.iterrows():
        out.append(
            {
                "season": season_slash,
                "stage": "relegation_playoff",
                "round": int(final_r + row["tm_playoff_spieltag"]),
                "team": row["team"],
                "points": row["points"],
                "position": row["rank"],
                "goal_diff": row["goal_diff"],
            }
        )
    return pd.DataFrame(out)


def collect_one_season(season_slash: str, dry_run: bool) -> tuple[pd.DataFrame | None, list[str]]:
    log: list[str] = []
    saison_id = int(season_slash.split("/")[0])
    mx = final_regular_round(season_slash)
    if mx is None:
        log.append(f"WARN {season_slash}: no matches file; skip")
        return None, log

    raw = scrape_transfermarkt_relegation_playoff_full_season(season_slash, saison_id)
    if raw is None or raw.empty:
        log.append(f"WARN {season_slash}: no relegation playoff on TM")
        return None, log

    first_st = raw["tm_playoff_spieltag"].min()
    n_playoff_teams = int(
        raw[raw["tm_playoff_spieltag"] == first_st]["team"].nunique()
    )
    df_out = build_output_for_season(season_slash, raw, mx)
    log.extend(validate_tables(raw, season_slash, n_playoff_teams))

    print("=" * 72)
    print(f"{season_slash}: relegation playoff teams = {n_playoff_teams} (expect 6-8)")
    print(f"  final_regular_round (matches max) = {int(mx)}")
    ccols = COLS_OUT
    first_tbl = df_out[df_out["round"] == df_out["round"].min()].sort_values(["position"])
    print("\nFirst playoff round table:")
    print(first_tbl[ccols].to_string(index=False))
    last_tbl = df_out[df_out["round"] == df_out["round"].max()].sort_values(
        ["position", "team"]
    )
    print("\nLast playoff round table:")
    print(last_tbl[ccols].to_string(index=False))

    n_spi = raw["tm_playoff_spieltag"].nunique()
    print(f"\nTM Spieltage scraped: {n_spi}")
    if log:
        print("Validation notes:")
        for w in log:
            print(" ", w)

    if df_out.groupby(["round", "team"]).size().max() > 1:
        raise ValueError(f"{season_slash}: duplicate team in same logical round")

    if dry_run:
        return df_out, log

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file = OUT_DIR / f'relegation_playoff_tracking_{season_slash.replace("/", "_")}.csv'
    backup_if_exists(per_file)
    df_out.sort_values(["round", "position", "team"]).to_csv(
        per_file, index=False, encoding="utf-8-sig"
    )
    print(f"Saved {per_file}")
    return df_out, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--season", help='one season label e.g. "2024/25"')
    args = parser.parse_args()
    seasons = [args.season] if args.season else SEASONS
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
        combined_path = OUT_DIR / "all_seasons_relegation_tracking.csv"
        backup_if_exists(combined_path)
        pd.concat(all_parts, ignore_index=True).sort_values(
            ["season", "round", "position", "team"]
        ).to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\nCombined: {combined_path}")


if __name__ == "__main__":
    main()
