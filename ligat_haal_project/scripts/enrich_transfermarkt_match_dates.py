from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
MATCHES_DIR = ROOT / "notebooks" / "data" / "matches"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
WEEKDAY_HE = {
    "Monday": "יום שני",
    "Tuesday": "יום שלישי",
    "Wednesday": "יום רביעי",
    "Thursday": "יום חמישי",
    "Friday": "יום שישי",
    "Saturday": "שבת",
    "Sunday": "יום ראשון",
}
SCORE_PATTERN = re.compile(r"^\d+\s*:\s*\d+$")
DATE_PATTERN = re.compile(r"(?P<weekday>[A-Za-z]{3})\s+(?P<date>\d{2}/\d{2}/\d{2})")


TEAM_NAME_MAP = {
    "Beitar Jerusalem": "B. Jerusalem",
    "B. Jerusalem": "B. Jerusalem",
    "Bnei Sakhnin": "Bnei Sakhnin",
    "Bnei Yehuda": "Bnei Yehuda",
    "Bnei Yehuda Tel Aviv": "Bnei Yehuda",
    "FC Ashdod": "FC Ashdod",
    "MS Ashdod": "FC Ashdod",
    "Hapoel Rishon LeZion": "H Rishon leZion",
    "H. Rishon leZion": "H Rishon leZion",
    "Hapoel Ashkelon": "H. Ashkelon",
    "Hapoel Be'er Sheva": "H. Beer Sheva",
    "Hapoel Beer Sheva": "H. Beer Sheva",
    "H. Beer Sheva": "H. Beer Sheva",
    "Hapoel Jerusalem": "H. Jerusalem",
    "H. Jerusalem": "H. Jerusalem",
    "Hapoel Kfar Saba": "H. Kfar Saba",
    "H. Kfar Saba": "H. Kfar Saba",
    "Hapoel Nof HaGalil": "H. Nof HaGalil",
    "H. Nof HaGalil": "H. Nof HaGalil",
    "Hapoel Petah Tikva": "H. Petah Tikva",
    "H. Petah Tikva": "H. Petah Tikva",
    "Hapoel Ramat Gan": "H. Ramat Gan",
    "H. Ramat Gan": "H. Ramat Gan",
    "Hakoah Amidar Ramat Gan": "Hakoah Amidar",
    "Hakoah Amidar": "Hakoah Amidar",
    "Hapoel Acre": "Hapoel Acre",
    "Hapoel Hadera": "Hapoel Hadera",
    "Hapoel Haifa": "Hapoel Haifa",
    "Hapoel Ra'anana": "Hapoel Raanana",
    "Hapoel Raanana": "Hapoel Raanana",
    "Hapoel Tel Aviv": "Hapoel Tel Aviv",
    "Ironi Tiberias": "Ironi Tiberias",
    "Ironi Kiryat Shmona": "Kiryat Shmona",
    "Kiryat Shmona": "Kiryat Shmona",
    "Maccabi Ahi Nazareth": "M. Ahi Nazareth",
    "M. Ahi Nazareth": "M. Ahi Nazareth",
    "Maccabi Bnei Reineh": "M. Bnei Reineh",
    "M. Bnei Reineh": "M. Bnei Reineh",
    "Maccabi Petah Tikva": "M. Petah Tikva",
    "M. Petah Tikva": "M. Petah Tikva",
    "Maccabi Tel Aviv": "M. Tel Aviv",
    "M. Tel Aviv": "M. Tel Aviv",
    "Maccabi Haifa": "Maccabi Haifa",
    "Maccabi Herzliya": "Maccabi Herzlya",
    "Maccabi Herzlya": "Maccabi Herzlya",
    "Maccabi Netanya": "Maccabi Netanya",
    "Sektzia Ness Ziona": "Ness Ziona",
    "Sekzia Ness Ziona": "Ness Ziona",
    "Ness Ziona": "Ness Ziona",
    "Hapoel Nir Ramat HaSharon": "Ramat haSharon",
    "Ramat HaSharon": "Ramat haSharon",
    "Ramat haSharon": "Ramat haSharon",
}


def season_file_path(season_year: int) -> Path:
    return MATCHES_DIR / f"matches_{season_year}_{str(season_year + 1)[-2:]}_ligat_haal_transfermarkt.csv"


def dated_season_file_path(season_year: int) -> Path:
    return MATCHES_DIR / f"matches_{season_year}_{str(season_year + 1)[-2:]}_ligat_haal_transfermarkt_dated.csv"


def fetch_transfermarkt_html(season_year: int) -> str:
    url = f"https://www.transfermarkt.com/ligat-haal/gesamtspielplan/wettbewerb/ISR1?saison_id={season_year}"
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    return response.text


def clean_team_name(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return TEAM_NAME_MAP.get(text, text)


def parse_date_header(value: str) -> tuple[str | None, str | None]:
    text = str(value).strip()
    if not text:
        return None, None
    match = DATE_PATTERN.search(text)
    if not match:
        return None, None
    date_part = match.group("date")
    remainder = text[match.end():].strip()
    time_part = remainder or None
    return date_part, time_part


def extract_date_text(value: str, fallback: str | None) -> str | None:
    text = str(value).strip()
    if not text:
        return fallback
    date_part, _ = parse_date_header(text)
    return date_part or fallback


def parse_transfermarkt_schedule(season_year: int) -> pd.DataFrame:
    html = fetch_transfermarkt_html(season_year)
    soup = BeautifulSoup(html, "html.parser")
    matches: list[dict[str, object]] = []
    round_counter = 0

    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")[:10]]
        if headers[:7] != ["Date", "Time", "Home team", "Home", "Result", "Away team", "Away"]:
            continue
        round_counter += 1

        current_date: str | None = None
        current_time: str | None = None

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue

            texts = [cell.get_text(" ", strip=True) for cell in cells]
            non_empty = [text for text in texts if text]
            if not non_empty:
                continue

            if len(non_empty) == 1 and "/" in non_empty[0]:
                current_date, current_time = parse_date_header(non_empty[0])
                continue

            if len(non_empty) == 1 and ":" in non_empty[0] and "/" not in non_empty[0]:
                current_time = non_empty[0]
                continue

            raw = texts + [""] * (7 - len(texts))
            score = raw[4].strip()
            if not SCORE_PATTERN.match(score):
                continue

            date_text = extract_date_text(raw[0], current_date)
            time_text = raw[1].strip() or current_time
            home = clean_team_name(raw[2])
            away = clean_team_name(raw[6])
            if not date_text or not home or not away:
                continue

            match_date = pd.to_datetime(date_text, format="%d/%m/%y", errors="coerce")
            if pd.isna(match_date):
                continue

            day_en = match_date.day_name()
            matches.append(
                {
                    "season": f"{season_year}/{str(season_year + 1)[-2:]}",
                    "season_year": season_year,
                    "round_verified": round_counter,
                    "home": home,
                    "score": score,
                    "away": away,
                    "match_date_raw": date_text,
                    "match_time_raw": time_text,
                    "match_date": match_date.strftime("%Y-%m-%d"),
                    "day_of_week_en": day_en,
                    "day_of_week_he": WEEKDAY_HE.get(day_en),
                    "day_of_week_num": int(match_date.dayofweek),
                }
            )

    schedule_df = pd.DataFrame(matches)
    if schedule_df.empty:
        raise RuntimeError(f"No matches parsed from Transfermarkt for {season_year}")

    schedule_df["dup_idx"] = schedule_df.groupby(["season", "home", "away", "score"]).cumcount()
    return schedule_df


def enrich_season_matches(season_year: int) -> tuple[pd.DataFrame, dict[str, object]]:
    season_path = season_file_path(season_year)
    existing_df = pd.read_csv(season_path)
    existing_df = existing_df.copy()
    existing_df = existing_df.rename(columns={"round": "round_original"})
    existing_df["match_sequence"] = range(1, len(existing_df) + 1)
    existing_df["dup_idx"] = existing_df.groupby(["season", "home", "away", "score"]).cumcount()

    scraped_df = parse_transfermarkt_schedule(season_year)
    scraped_df = scraped_df.copy()
    scraped_df["match_sequence"] = range(1, len(scraped_df) + 1)

    enriched_df = existing_df.merge(
        scraped_df[
            [
                "season",
                "round_verified",
                "home",
                "away",
                "score",
                "dup_idx",
                "match_date_raw",
                "match_time_raw",
                "match_date",
                "day_of_week_en",
                "day_of_week_he",
                "day_of_week_num",
            ]
        ],
        on=["season", "home", "away", "score", "dup_idx"],
        how="left",
    )

    if enriched_df["match_date"].isna().any():
        sequence_fallback_df = scraped_df.head(len(existing_df)).copy()
        sequence_fallback_df = sequence_fallback_df[
            [
                "match_sequence",
                "round_verified",
                "match_date_raw",
                "match_time_raw",
                "match_date",
                "day_of_week_en",
                "day_of_week_he",
                "day_of_week_num",
            ]
        ].rename(
            columns={
                "round_verified": "round_verified_seq",
                "match_date_raw": "match_date_raw_seq",
                "match_time_raw": "match_time_raw_seq",
                "match_date": "match_date_seq",
                "day_of_week_en": "day_of_week_en_seq",
                "day_of_week_he": "day_of_week_he_seq",
                "day_of_week_num": "day_of_week_num_seq",
            }
        )
        enriched_df = enriched_df.merge(sequence_fallback_df, on="match_sequence", how="left")
        enriched_df["round_verified"] = enriched_df["round_verified"].fillna(enriched_df["round_verified_seq"])
        for target_col, fallback_col in [
            ("match_date_raw", "match_date_raw_seq"),
            ("match_time_raw", "match_time_raw_seq"),
            ("match_date", "match_date_seq"),
            ("day_of_week_en", "day_of_week_en_seq"),
            ("day_of_week_he", "day_of_week_he_seq"),
            ("day_of_week_num", "day_of_week_num_seq"),
        ]:
            enriched_df[target_col] = enriched_df[target_col].fillna(enriched_df[fallback_col])
        enriched_df = enriched_df.drop(columns=[
            "round_verified_seq",
            "match_date_raw_seq",
            "match_time_raw_seq",
            "match_date_seq",
            "day_of_week_en_seq",
            "day_of_week_he_seq",
            "day_of_week_num_seq",
        ])

    missing_dates = int(enriched_df["match_date"].isna().sum())
    summary = {
        "season": f"{season_year}/{str(season_year + 1)[-2:]}",
        "source_rows": int(len(existing_df)),
        "scraped_rows": int(len(scraped_df)),
        "verified_rounds": int(pd.Series(scraped_df["round_verified"]).nunique()),
        "missing_dates": missing_dates,
        "coverage_pct": float((1 - missing_dates / len(existing_df)) * 100 if len(existing_df) else 0.0),
    }
    if len(scraped_df) < len(existing_df):
        raise RuntimeError(
            f"Row count mismatch for {summary['season']}: existing={len(existing_df)}, scraped={len(scraped_df)}"
        )
    if missing_dates:
        missing_preview = enriched_df.loc[enriched_df["match_date"].isna(), ["home", "score", "away"]].head(10)
        raise RuntimeError(f"Missing dates remain for {summary['season']}\n{missing_preview.to_string(index=False)}")

    if enriched_df["round_verified"].isna().any():
        raise RuntimeError(f"Missing verified rounds remain for {summary['season']}")

    enriched_df["round_verified"] = enriched_df["round_verified"].astype(int)
    enriched_df["round"] = enriched_df["round_verified"]
    enriched_df = enriched_df.drop(columns=["dup_idx", "match_sequence"])
    ordered_columns = [
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
    enriched_df = enriched_df[ordered_columns]
    return enriched_df, summary


def main() -> None:
    season_files = sorted(MATCHES_DIR.glob("matches_????_??_ligat_haal_transfermarkt.csv"))
    season_years = sorted(int(path.stem.split("_")[1]) for path in season_files)

    all_enriched: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for season_year in season_years:
        print(f"Enriching {season_year}/{str(season_year + 1)[-2:]}...")
        enriched_df, summary = enrich_season_matches(season_year)
        enriched_df.to_csv(dated_season_file_path(season_year), index=False, encoding="utf-8-sig")
        all_enriched.append(enriched_df)
        coverage_rows.append(summary)
        print(
            f"  saved {dated_season_file_path(season_year).name} | rows={summary['source_rows']} | coverage={summary['coverage_pct']:.1f}%"
        )
        time.sleep(1)

    combined_df = pd.concat(all_enriched, ignore_index=True)
    combined_output = MATCHES_DIR / "matches_all_seasons_ligat_haal_transfermarkt_dated.csv"
    combined_df.to_csv(combined_output, index=False, encoding="utf-8-sig")

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_output = MATCHES_DIR / "matches_all_seasons_ligat_haal_transfermarkt_date_coverage_summary.csv"
    coverage_df.to_csv(coverage_output, index=False, encoding="utf-8-sig")

    print("Saved outputs:")
    print(combined_output)
    print(coverage_output)


if __name__ == "__main__":
    main()