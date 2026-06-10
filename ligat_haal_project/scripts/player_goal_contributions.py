"""
Player goals+assists metrics from Transfermarkt top scorers / assists lists.

Used by build_research_summary_table.py for season-level 15+ G+A columns.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PLAYER_STATS_DIR = ROOT / "notebooks" / "data" / "raw" / "player_stats"
OUTPUT = ROOT / "notebooks" / "outputs" / "research_summary"

GOAL_CONTRIBUTION_THRESHOLD = 15
MERGE_KEY = ["player_url", "season_year", "club"]

TM_BASE = "https://www.transfermarkt.com"
SCORERS_LIST_URL = TM_BASE + "/ligat-ha-al/torschuetzenliste/wettbewerb/ISR1/saison_id/{season}"
ASSISTS_URL = TM_BASE + "/ligat-ha-al/assistliste/wettbewerb/ISR1/saison_id/{season}"
SCORERS_TAIL_LAYOUT = "matches_goals"

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15",
]


def season_start_year(season: str) -> int:
    m = re.match(r"(\d{4})/(\d{2})", season.strip())
    if not m:
        raise ValueError(f"Invalid season label: {season}")
    return int(m.group(1))


def season_file_slug(season: str) -> str:
    m = re.match(r"(\d{4})/(\d{2})", season.strip())
    if not m:
        return ""
    return f"{m.group(1)}_{m.group(2)}"


def scorers_path_for_season(season: str) -> Path:
    return PLAYER_STATS_DIR / f"top_scorers_{season_file_slug(season)}_ligat_haal_transfermarkt.csv"


def assists_path_for_season(season: str) -> Path:
    return PLAYER_STATS_DIR / f"top_assists_{season_file_slug(season)}_ligat_haal_transfermarkt.csv"


# ---------------------------------------------------------------------------
# HTTP + Transfermarkt scraper (from player_stats collection notebook)
# ---------------------------------------------------------------------------
def http_get(url: str, retries: int = 3, timeout: int = 30) -> str:
    last_err: Exception | None = None
    sess = requests.Session()
    for attempt in range(1, retries + 1):
        hdrs = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = sess.get(url, headers=hdrs, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_err = exc
            time.sleep(0.8 * attempt)
    raise last_err  # type: ignore[misc]


def _norm_header(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ").strip()).lower()


def _thead_labels(table) -> list[str]:
    thead = table.find("thead")
    if not thead:
        return []
    rows = thead.find_all("tr")
    best: list[str] = []
    for tr in rows:
        labels = [_norm_header(th.get_text(" ", strip=True)) for th in tr.find_all(["th", "td"])]
        if len(labels) > len(best):
            best = labels
    return best


def _cell_player_name(td) -> tuple[str, str]:
    a = td.find("a", class_="hauptlink") or td.find("a", href=re.compile(r"/profil/spieler/"))
    name = a.get_text(strip=True) if a else td.get_text(strip=True)
    href = ""
    if a:
        h = a.get("href", "")
        href = TM_BASE + h if h.startswith("/") else h
    return name, href


def _player_cell_full(td) -> tuple[str, str, str]:
    position = ""
    inline = td.find("table", class_="inline-table")
    name, url = "", ""
    if inline:
        trs = inline.find_all("tr")
        if trs:
            a = trs[0].find("a", class_="hauptlink") or trs[0].find("a", href=re.compile(r"/profil/spieler/"))
            if a:
                name = a.get_text(strip=True)
                h = a.get("href", "")
                url = TM_BASE + h if h.startswith("/") else h
            if len(trs) > 1:
                position = trs[1].get_text(strip=True)
    if not name:
        name, url = _cell_player_name(td)
    return name, url, position


def _nat_cell(td) -> str:
    imgs = td.find_all("img", class_="flaggenrahmen")
    parts = []
    for img in imgs:
        t = (img.get("title") or img.get("alt") or "").strip()
        if t:
            parts.append(t)
    return "; ".join(parts)


def _cell_club_name(td) -> tuple[str, str]:
    a = td.find("a", class_="hauptlink") or td.find("a", href=re.compile(r"/startseite/verein/"))
    title = (a.get("title") or "").strip() if a else ""
    name = title or (a.get_text(strip=True) if a else td.get_text(strip=True))
    href = ""
    if a:
        h = a.get("href", "")
        href = TM_BASE + h if h.startswith("/") else h
    return name, href


def _pick_column_index(headers: list[str], keywords: tuple[str, ...]) -> int | None:
    for i, h in enumerate(headers):
        for kw in keywords:
            if kw in h:
                return i
    return None


def _parse_int_text(t: str) -> int | None:
    t = t.strip().replace(".", "").replace(",", "")
    if not t or t == "-":
        return None
    m = re.search(r"-?\d+", t)
    return int(m.group(0)) if m else None


def _resolve_tail_indices(headers: list[str], n_cells: int, stat_mode: str) -> dict:
    hl = [_norm_header(h) for h in headers]
    while len(hl) < n_cells:
        hl.append("")
    matches = _pick_column_index(hl, ("match", "spiele", "spiel", "games"))
    goals = _pick_column_index(hl, ("goals", "tore"))
    assists = _pick_column_index(hl, ("assists", "vorlag"))

    if n_cells < 7:
        return {"matches": matches, "goals": goals, "assists": assists}

    if stat_mode == "assists":
        return {
            "matches": matches if matches is not None else 5,
            "goals": goals,
            "assists": assists if assists is not None else 6,
        }

    if goals is not None and assists is not None:
        return {"matches": matches, "goals": goals, "assists": assists}
    if matches is not None and goals is not None:
        return {"matches": matches, "goals": goals, "assists": assists}

    if SCORERS_TAIL_LAYOUT == "goals_assists":
        return {"matches": matches, "goals": 5, "assists": 6}
    return {
        "matches": matches if matches is not None else 5,
        "goals": goals if goals is not None else 6,
        "assists": assists,
    }


def _page_has_stat_columns(headers: list[str], stat_mode: str) -> bool:
    hl = [_norm_header(h) for h in headers]
    if stat_mode == "assists":
        if any(k in h for h in hl for k in ("assists", "vorlag")):
            return True
        # Older TM pages list players without assist counts (market value only).
        return len(hl) >= 6 and any(k in h for h in hl for k in ("match", "spiele", "spiel", "games"))
    if any(k in h for h in hl for k in ("goals", "tore")):
        return True
    return len(hl) >= 6


def parse_items_table_page(html: str, stat_mode: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        return pd.DataFrame()

    headers = _thead_labels(table)
    if not _page_has_stat_columns(headers, stat_mode):
        return pd.DataFrame()
    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows_out: list[dict[str, Any]] = []
    tail_ix = None
    for tr in tbody.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 4:
            continue
        if tail_ix is None:
            tail_ix = _resolve_tail_indices(headers, len(tds), stat_mode)

        rank_t = tds[0].get_text(strip=True)
        if not rank_t.isdigit():
            continue

        player, player_url, position = _player_cell_full(tds[1]) if len(tds) > 1 else ("", "", "")
        if not player:
            continue

        nationality = _nat_cell(tds[2]) if len(tds) > 2 else ""
        age = _parse_int_text(tds[3].get_text(strip=True)) if len(tds) > 3 else None

        club, club_url = ("", "")
        if len(tds) > 4:
            club, club_url = _cell_club_name(tds[4])
            if not club.strip():
                club = tds[4].get_text(" ", strip=True)

        mi = tail_ix.get("matches")
        gi = tail_ix.get("goals")
        ai = tail_ix.get("assists")

        matches = _parse_int_text(tds[mi].get_text(strip=True)) if mi is not None and mi < len(tds) else None
        goals = _parse_int_text(tds[gi].get_text(strip=True)) if gi is not None and gi < len(tds) else None
        assists = _parse_int_text(tds[ai].get_text(strip=True)) if ai is not None and ai < len(tds) else None

        if stat_mode == "goals" and goals is None and len(tds) >= 1:
            goals = _parse_int_text(tds[-1].get_text(strip=True))
        if stat_mode == "goals" and matches is None and len(tds) >= 2:
            matches = _parse_int_text(tds[-2].get_text(strip=True))
        if stat_mode == "assists" and assists is None and len(tds) >= 1:
            assists = _parse_int_text(tds[-1].get_text(strip=True))
        if stat_mode == "assists" and matches is None and len(tds) >= 2:
            matches = _parse_int_text(tds[-2].get_text(strip=True))

        rows_out.append(
            {
                "rank": int(rank_t),
                "player": player,
                "player_url": player_url,
                "position": position,
                "nationality": nationality,
                "age": age,
                "club": club,
                "club_url": club_url,
                "matches": matches,
                "goals": goals,
                "assists": assists,
            }
        )

    if not rows_out:
        return pd.DataFrame()

    df = pd.DataFrame(rows_out)
    if stat_mode == "assists":
        return df[
            [
                "rank", "player", "player_url", "position", "nationality", "age",
                "club", "club_url", "matches", "assists", "goals",
            ]
        ]
    return df[
        [
            "rank", "player", "player_url", "position", "nationality", "age",
            "club", "club_url", "matches", "goals", "assists",
        ]
    ]


def fetch_all_pages(season_year: int, url_template: str, stat_mode: str, pause: float = 1.0) -> pd.DataFrame:
    base = url_template.format(season=season_year)
    all_parts: list[pd.DataFrame] = []
    page = 1
    while page <= 40:
        url = base if page == 1 else f"{base}/page/{page}"
        html = http_get(url)
        df = parse_items_table_page(html, stat_mode=stat_mode)
        if df.empty:
            break
        all_parts.append(df)
        if len(df) < 25:
            break
        page += 1
        time.sleep(pause)
    if not all_parts:
        return pd.DataFrame()
    return pd.concat(all_parts, ignore_index=True)


def scrape_top_scorers_season(season_year: int) -> pd.DataFrame | None:
    season_str = f"{season_year}/{str(season_year + 1)[-2:]}"
    print(f"  Scraping scorers {season_str}...")
    try:
        df = fetch_all_pages(season_year, SCORERS_LIST_URL, "goals")
        if df.empty:
            return None
        df.insert(0, "season", season_str)
        df.insert(1, "season_year", season_year)
        return df
    except Exception as exc:
        print(f"  Scorers scrape failed {season_str}: {exc}")
        return None


def scrape_top_assists_season(season_year: int) -> pd.DataFrame | None:
    season_str = f"{season_year}/{str(season_year + 1)[-2:]}"
    print(f"  Scraping assists {season_str}...")
    try:
        df = fetch_all_pages(season_year, ASSISTS_URL, "assists")
        if df.empty:
            return None
        df.insert(0, "season", season_str)
        df.insert(1, "season_year", season_year)
        return df
    except Exception as exc:
        print(f"  Assists scrape failed {season_str}: {exc}")
        return None


def _save_player_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def is_valid_stats_df(df: pd.DataFrame, value_col: str) -> bool:
    if df.empty or value_col not in df.columns:
        return False
    vals = pd.to_numeric(df[value_col], errors="coerce")
    if vals.notna().sum() == 0:
        return False
    if vals.max() > 35:
        return False
    if "club" in df.columns:
        clubs = df["club"].astype(str).str.strip().replace({"nan": "", "None": ""})
        if (clubs != "").mean() < 0.25:
            return False
    return True


def is_valid_stats_file(path: Path, value_col: str) -> bool:
    if not path.exists() or path.stat().st_size < 40:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or value_col not in df.columns:
        return False
    vals = pd.to_numeric(df[value_col], errors="coerce")
    if vals.notna().sum() == 0:
        return False
    if vals.max() > 35:
        return False
    if "club" in df.columns:
        clubs = df["club"].astype(str).str.strip().replace({"nan": "", "None": ""})
        if (clubs != "").mean() < 0.25:
            return False
    return True


def scrape_season_player_stats(season: str) -> bool:
    year = season_start_year(season)
    scorers = scrape_top_scorers_season(year)
    time.sleep(1.0)
    assists = scrape_top_assists_season(year)
    ok = True
    if scorers is not None and not scorers.empty:
        _save_player_csv(scorers, scorers_path_for_season(season))
        print(f"  Saved scorers: {scorers_path_for_season(season).name} ({len(scorers)} rows)")
    else:
        ok = False
    if assists is not None and is_valid_stats_df(assists, "assists"):
        _save_player_csv(assists, assists_path_for_season(season))
        print(f"  Saved assists: {assists_path_for_season(season).name} ({len(assists)} rows)")
    elif assists is not None and not assists.empty:
        print(f"  Assists scrape for {season} unusable (layout/values); using scorers only.")
    else:
        print(f"  No assists data on Transfermarkt for {season}; using scorers only.")
    return scorers is not None and not scorers.empty


def ensure_player_stats_for_seasons(seasons: list[str], scrape_if_missing: bool = True) -> list[str]:
    """Return list of seasons that were (re)scraped."""
    scraped: list[str] = []
    PLAYER_STATS_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        sc_path = scorers_path_for_season(season)
        as_path = assists_path_for_season(season)
        need_scorers = not is_valid_stats_file(sc_path, "goals")
        need_assists = not is_valid_stats_file(as_path, "assists")
        if not need_scorers and not need_assists:
            continue
        if not scrape_if_missing:
            continue
        print(f"Player stats missing/invalid for {season} — scraping Transfermarkt...")
        if scrape_season_player_stats(season):
            scraped.append(season)
    if scraped:
        rebuild_all_seasons_files()
    return scraped


def rebuild_all_seasons_files() -> None:
    scorer_parts: list[pd.DataFrame] = []
    assist_parts: list[pd.DataFrame] = []
    for sc_path in sorted(PLAYER_STATS_DIR.glob("top_scorers_*_ligat_haal_transfermarkt.csv")):
        if "all_seasons" in sc_path.name:
            continue
        scorer_parts.append(pd.read_csv(sc_path))
    for as_path in sorted(PLAYER_STATS_DIR.glob("top_assists_*_ligat_haal_transfermarkt.csv")):
        if "all_seasons" in as_path.name:
            continue
        assist_parts.append(pd.read_csv(as_path))
    if scorer_parts:
        _save_player_csv(
            pd.concat(scorer_parts, ignore_index=True),
            PLAYER_STATS_DIR / "top_scorers_all_seasons_ligat_haal_transfermarkt.csv",
        )
    if assist_parts:
        _save_player_csv(
            pd.concat(assist_parts, ignore_index=True),
            PLAYER_STATS_DIR / "top_assists_all_seasons_ligat_haal_transfermarkt.csv",
        )


# ---------------------------------------------------------------------------
# Merge + metrics
# ---------------------------------------------------------------------------
def _empty_assists_frame(scorers: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "season", "season_year", "player", "player_url", "club",
            "goals_as", "assists_as",
        ]
    )


def load_merged_season_stats(scorers_path: Path, assists_path: Path | None = None) -> pd.DataFrame:
    def _dedupe(df: pd.DataFrame, goals_c: str, ast_c: str) -> pd.DataFrame:
        d = df.copy()
        if "club" in d.columns:
            d["club"] = d["club"].astype(str).replace({"nan": "", "None": "", "<NA>": ""})
        d[goals_c] = pd.to_numeric(d[goals_c], errors="coerce")
        d[ast_c] = pd.to_numeric(d[ast_c], errors="coerce")
        agg: dict[str, str] = {"season": "first", "player": "first", goals_c: "max", ast_c: "max"}
        if "rank" in d.columns:
            agg["rank"] = "min"
        return d.groupby(MERGE_KEY, as_index=False, dropna=False).agg(agg)

    s = _dedupe(
        pd.read_csv(scorers_path).rename(columns={"goals": "goals_sc", "assists": "assists_sc"}),
        "goals_sc",
        "assists_sc",
    )[["season", "season_year", "player", "player_url", "club", "goals_sc", "assists_sc"]]
    if assists_path is not None and is_valid_stats_file(assists_path, "assists"):
        a = _dedupe(
            pd.read_csv(assists_path).rename(columns={"goals": "goals_as", "assists": "assists_as"}),
            "goals_as",
            "assists_as",
        )[["season", "season_year", "player", "player_url", "club", "goals_as", "assists_as"]]
    else:
        a = _empty_assists_frame(s)

    m = s.merge(a, on=MERGE_KEY, how="outer", suffixes=("_s", "_a"))
    m["player"] = m["player_s"].fillna(m["player_a"])
    m["season"] = m["season_s"].fillna(m["season_a"])
    m = m.drop(columns=["player_s", "player_a", "season_s", "season_a"])

    for col in ("goals_sc", "assists_sc", "goals_as", "assists_as"):
        m[col] = pd.to_numeric(m[col], errors="coerce")

    m["goals"] = m[["goals_sc", "goals_as"]].max(axis=1, skipna=True)
    m["assists"] = m[["assists_sc", "assists_as"]].max(axis=1, skipna=True)
    m["goals"] = m["goals"].fillna(0).astype(int)
    m["assists"] = m["assists"].fillna(0).astype(int)
    m["goals_plus_assists"] = m["goals"] + m["assists"]
    return m


def load_all_merged_player_stats(seasons: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for season in seasons:
        sc = scorers_path_for_season(season)
        if not is_valid_stats_file(sc, "goals"):
            continue
        as_ = assists_path_for_season(season)
        assists_arg = as_ if is_valid_stats_file(as_, "assists") else None
        parts.append(load_merged_season_stats(sc, assists_arg))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _valid_club_series(clubs: pd.Series) -> pd.Series:
    return clubs.astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})


def compute_goal_contribution_summary(
    seasons: list[str],
    threshold: int = GOAL_CONTRIBUTION_THRESHOLD,
    scrape_if_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (season_summary, player_audit) for players with goals+assists >= threshold.
    """
    ensure_player_stats_for_seasons(seasons, scrape_if_missing=scrape_if_missing)
    merged = load_all_merged_player_stats(seasons)

    summary_rows: list[dict[str, Any]] = []
    for season in seasons:
        if merged.empty or season not in merged["season"].values:
            summary_rows.append(
                {
                    "season": season,
                    "players_with_15_plus_goals_assists": pd.NA,
                    "distinct_teams_with_15_plus_goals_assists": pd.NA,
                    "player_stats_available": False,
                }
            )
            continue
        sub = merged[merged["season"] == season]
        above = sub[sub["goals_plus_assists"] >= threshold].copy()
        clubs = _valid_club_series(above["club"])
        valid_clubs = clubs[clubs != ""]
        summary_rows.append(
            {
                "season": season,
                "players_with_15_plus_goals_assists": int(above["player_url"].nunique()),
                "distinct_teams_with_15_plus_goals_assists": int(valid_clubs.nunique()) if len(valid_clubs) else 0,
                "player_stats_available": True,
            }
        )

    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame()
    if not merged.empty:
        audit = merged[merged["goals_plus_assists"] >= threshold][
            ["season", "club", "player", "goals", "assists", "goals_plus_assists"]
        ].sort_values(["season", "goals_plus_assists", "goals"], ascending=[True, False, False])
    return summary, audit


def apply_goal_contribution_to_summary(
    summary: pd.DataFrame,
    seasons: list[str] | None = None,
    scrape_if_missing: bool = True,
) -> pd.DataFrame:
    season_list = seasons or summary["season"].tolist()
    ga_summary, audit = compute_goal_contribution_summary(
        season_list, scrape_if_missing=scrape_if_missing
    )
    out = summary.drop(
        columns=[
            c
            for c in (
                "players_with_15_plus_goals_assists",
                "distinct_teams_with_15_plus_goals_assists",
                "player_stats_available",
            )
            if c in summary.columns
        ],
        errors="ignore",
    )
    out = out.merge(ga_summary, on="season", how="left")
    if not audit.empty:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        audit.to_csv(
            OUTPUT / "players_15_plus_goals_assists_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return out
