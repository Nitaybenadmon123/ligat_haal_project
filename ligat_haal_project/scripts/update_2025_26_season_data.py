#!/usr/bin/env python3
"""Scrape and refresh 2025/26 match + tracking data from Transfermarkt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "data"
MATCHES = DATA / "matches"
INTERIM = DATA / "interim" / "scraped_standings"
SEASON = "2025/26"
SEASON_YEAR = 2025

sys.path.insert(0, str(ROOT / "scripts"))
from enrich_transfermarkt_match_dates import clean_team_name, parse_transfermarkt_schedule  # noqa: E402


def save_matches() -> None:
    scraped = parse_transfermarkt_schedule(SEASON_YEAR)
    scraped["home"] = scraped["home"].map(clean_team_name)
    scraped["away"] = scraped["away"].map(clean_team_name)

    basic = scraped[
        ["round_verified", "home", "score", "away", "season", "season_year"]
    ].rename(columns={"round_verified": "round"})
    basic_path = MATCHES / "matches_2025_26_ligat_haal_transfermarkt.csv"
    basic.to_csv(basic_path, index=False, encoding="utf-8-sig")
    print(f"Saved {basic_path.name}: {len(basic)} matches, max round {int(basic['round'].max())}")

    dated = scraped.copy()
    dated["round"] = dated["round_verified"]
    dated["round_original"] = range(1, len(dated) + 1)
    dated = dated[
        [
            "round",
            "round_verified",
            "round_original",
            "home",
            "score",
            "away",
            "season",
            "season_year",
            "match_date_raw",
            "match_time_raw",
            "match_date",
            "day_of_week_en",
            "day_of_week_he",
            "day_of_week_num",
        ]
    ]
    dated_path = MATCHES / "matches_2025_26_ligat_haal_transfermarkt_dated.csv"
    dated.to_csv(dated_path, index=False, encoding="utf-8-sig")
    print(f"Saved {dated_path.name}")

    for all_name in (
        "matches_all_seasons_ligat_haal_transfermarkt_dated.csv",
        "matches_all_seasons_ligat_haal_transfermarkt.csv",
    ):
        all_path = MATCHES / all_name
        if not all_path.exists():
            continue
        all_df = pd.read_csv(all_path)
        all_df = all_df[all_df["season"] != SEASON]
        chunk = dated if "dated" in all_name else basic
        all_df = pd.concat([all_df, chunk], ignore_index=True)
        all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        print(f"Updated {all_name}: {len(all_df)} total rows")


def run_collector(script_name: str) -> None:
    script = ROOT / "notebooks" / "relegation_battles" / script_name
    print(f"\nRunning {script_name}...")
    subprocess.run(
        [sys.executable, str(script), "--season", SEASON],
        check=True,
        cwd=str(ROOT),
    )


def augment_regular_tracking_rounds_25_26() -> None:
    """TM collector can stop at round 24 due to duplicate-snapshot guard; fetch 25-26 explicitly."""
    import time

    import requests
    from bs4 import BeautifulSoup

    path = DATA / "regular_season_tracking" / "regular_season_tracking_2025_26.csv"
    if not path.exists():
        return
    base = pd.read_csv(path)
    if int(base["round"].max()) >= 26:
        return

    def scrape_spieltag(sp: int) -> pd.DataFrame:
        url = (
            "https://www.transfermarkt.com/ligat-haal/spieltagtabelle/wettbewerb/ISR1"
            f"?saison_id={SEASON_YEAR}&spieltag={sp}"
        )
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"},
            timeout=25,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", class_="items")
        rows: list[dict] = []
        if table is None or table.find("tbody") is None:
            return pd.DataFrame(rows)
        for tr in table.find("tbody").find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 10:
                continue
            rows.append(
                {
                    "season": SEASON,
                    "stage": "regular",
                    "round": sp,
                    "team": cells[2].text.strip(),
                    "points": int(cells[9].text.strip()),
                    "position": int(cells[0].text.strip()),
                    "goal_diff": int(cells[8].text.strip().replace("\u2212", "-")),
                }
            )
        return pd.DataFrame(rows)

    extra = []
    for sp in range(int(base["round"].max()) + 1, 27):
        time.sleep(0.2)
        chunk = scrape_spieltag(sp)
        if not chunk.empty:
            extra.append(chunk)
    if not extra:
        return
    out = pd.concat([base, *extra], ignore_index=True).sort_values(["round", "position"])
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Augmented {path.name} to max round {int(out['round'].max())}")


def build_final_tables() -> None:
    reg_track = DATA / "regular_season_tracking" / "regular_season_tracking_2025_26.csv"
    cp_track = DATA / "championship_playoff_tracking" / "championship_playoff_tracking_2025_26.csv"

    if reg_track.exists():
        reg = pd.read_csv(reg_track)
        last_r = int(reg["round"].max())
        last = reg[reg["round"] == last_r].sort_values("position")
        final = pd.DataFrame(
            {
                "rank": last["position"].astype(int),
                "team": last["team"],
                "points": last["points"].astype(int),
                "goal_diff": last["goal_diff"].astype(int),
            }
        )
        out = INTERIM / "regular_final_tables" / "regular_final_table_2025_26.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved {out.relative_to(ROOT)}")

    if cp_track.exists():
        cp = pd.read_csv(cp_track)
        last_r = int(cp["round"].max())
        last = cp[cp["round"] == last_r].sort_values("position")
        reg_max = 26
        if reg_track.exists():
            reg_max = int(pd.read_csv(reg_track)["round"].max())
        final = pd.DataFrame(
            {
                "rank": last["position"].astype(int),
                "team": last["team"],
                "points": last["points"].astype(int),
                "season": SEASON,
                "playoff_round": int(last_r - reg_max),
            }
        )
        out = INTERIM / "playoff_final_tables" / "playoff_final_table_2025_26.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved {out.relative_to(ROOT)}")


def main() -> None:
    print("=== 1. Scrape matches from Transfermarkt ===")
    save_matches()

    print("\n=== 2. Scrape round-by-round tracking ===")
    run_collector("collect_regular_season_tracking.py")
    run_collector("collect_championship_playoff_tracking.py")
    run_collector("collect_relegation_playoff_tracking.py")
    augment_regular_tracking_rounds_25_26()

    print("\n=== 3. Build final standings tables ===")
    build_final_tables()

    print("\nDone. Run build_research_summary_table.py to refresh outputs.")


if __name__ == "__main__":
    main()
