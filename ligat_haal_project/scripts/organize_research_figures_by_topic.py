"""
Copy all research figures into topic folders for report writing.

Run: python scripts/organize_research_figures_by_topic.py

Original files are left in place; this script creates a curated copy tree under:
  notebooks/outputs/research_figures_by_topic/
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "outputs" / "research_figures_by_topic"

# Topic folders (numbered for writing order)
TOPICS: dict[str, str] = {
    "01_title_race": "Title race competitiveness, leadership changes, points gaps",
    "02_relegation_battles": "Relegation zone volatility, survival battles, closeness",
    "03_match_outcomes_and_goals": "Home/draw/away rates and goals per match",
    "04_league_balance_and_structure": "League balance, Gini, structural comparisons",
    "05_playoffs_vs_regular_season": "Regular season vs playoff differences",
    "06_attendance_overview": "Season/team attendance levels, trends, forecasts",
    "07_attendance_by_match_day": "Attendance by weekday (scheduling impact on crowds)",
    "08_match_scheduling_weekdays": "When matches are played (volume, not attendance)",
    "09_demographics_and_club_context": "Population, budget, attendance efficiency by club",
    "10_economic_data_and_squad_values": "Budgets, squad values, foreign players",
    "11_league_dynamics_and_misc": "Rank changes, stability, data-quality checks",
}

# Explicit filename -> topic (basename only)
EXPLICIT: dict[str, str] = {
    # build_research_summary_table.py — title race
    "final_gap_1st_2nd_by_season.png": "01_title_race",
    "title_decision_round_by_season.png": "01_title_race",
    "lead_changes_after_round_10_by_season.png": "01_title_race",
    "teams_in_title_race_5_rounds_before_end.png": "01_title_race",
    "champion_leadership_percentage_by_season.png": "01_title_race",
    "title_race_closeness_score_by_season.png": "01_title_race",
    "first_round_champion_was_1st_distribution.png": "01_title_race",
    "title_race_timeline_most_competitive_2010_11.png": "01_title_race",
    "title_race_timeline_least_competitive_2018_19.png": "01_title_race",
    # relegation
    "relegation_zone_changes_by_season.png": "02_relegation_battles",
    "teams_in_relegation_race_last_5_rounds.png": "02_relegation_battles",
    "relegation_closeness_score_by_season.png": "02_relegation_battles",
    "relegation_metrics_charts.png": "02_relegation_battles",
    "viz1_survival_gap_trend.png": "02_relegation_battles",
    "viz2_zone_volatility_bars.png": "02_relegation_battles",
    "viz3_contenders_per_season.png": "02_relegation_battles",
    "viz4_closeness_vs_volatility_scatter.png": "02_relegation_battles",
    "viz5_bonus_rank_paths_2022_23.png": "02_relegation_battles",
    # match outcomes
    "home_draw_away_percentages_by_season.png": "03_match_outcomes_and_goals",
    "average_goals_per_match_by_season.png": "03_match_outcomes_and_goals",
    # league balance
    "league_balance_score_by_season.png": "04_league_balance_and_structure",
    "gini_coefficient_trend.png": "04_league_balance_and_structure",
    "league_structure_comparison.png": "04_league_balance_and_structure",
    "league_structure_distributions.png": "04_league_balance_and_structure",
    "top_2_market_share.png": "04_league_balance_and_structure",
    "engagement_score.png": "04_league_balance_and_structure",
    # playoffs vs regular
    "regular_vs_playoff_goals.png": "05_playoffs_vs_regular_season",
    "regular_vs_playoff_home_win_percentage.png": "05_playoffs_vs_regular_season",
    "leadership_changes_regular_vs_playoff.png": "05_playoffs_vs_regular_season",
    "points_gap_regular_vs_playoff_comparison.png": "05_playoffs_vs_regular_season",
    "points_gap_regular_vs_playoff_comparison_FIXED.png": "05_playoffs_vs_regular_season",
    "combined_regular_playoff_gaps.png": "05_playoffs_vs_regular_season",
    "halving_rule_effect_2009_10.png": "05_playoffs_vs_regular_season",
    # attendance overview
    "attendance_boxplot.png": "06_attendance_overview",
    "attendance_by_season.png": "06_attendance_overview",
    "attendance_distribution.png": "06_attendance_overview",
    "attendance_heatmap.png": "06_attendance_overview",
    "attendance_trends_all_teams.png": "06_attendance_overview",
    "attendance_trends_bottom3_decline.png": "06_attendance_overview",
    "attendance_trends_distribution.png": "06_attendance_overview",
    "attendance_trends_top3_growth.png": "06_attendance_overview",
    "attendance_trends_top_bottom.png": "06_attendance_overview",
    "capacity_vs_attendance.png": "06_attendance_overview",
    "spectators_trend.png": "06_attendance_overview",
    "top_teams_attendance.png": "06_attendance_overview",
    "top_teams_trends.png": "06_attendance_overview",
    "forecast_distribution.png": "06_attendance_overview",
    "forecast_league_average.png": "06_attendance_overview",
    "forecast_top_10_teams.png": "06_attendance_overview",
    "attendance_by_day_of_week.png": "07_attendance_by_match_day",
    # attendance day analysis (script)
    "01_attendance_boxplot_by_weekday.png": "07_attendance_by_match_day",
    "02_attendance_mean_vs_median_by_weekday.png": "07_attendance_by_match_day",
    "03_attendance_match_scatter_by_weekday.png": "07_attendance_by_match_day",
    "04_attendance_reliable_days_only.png": "07_attendance_by_match_day",
    "05_attendance_share_vs_match_share_by_weekday.png": "07_attendance_by_match_day",
    "06_saturday_vs_weekday_attendance_trend.png": "07_attendance_by_match_day",
    "07_attendance_mean_ci_by_weekday.png": "07_attendance_by_match_day",
    "08_top_attendance_matches_by_weekday.png": "07_attendance_by_match_day",
    # weekday scheduling (match volume)
    "weekday_concentration_and_weekend_exposure.png": "08_match_scheduling_weekdays",
    "weekday_distribution_boxplot.png": "08_match_scheduling_weekdays",
    "weekday_metric_correlations_heatmap.png": "08_match_scheduling_weekdays",
    "weekday_phase_mix.png": "08_match_scheduling_weekdays",
    "weekday_round_day_span_histogram.png": "08_match_scheduling_weekdays",
    "weekday_season_heatmap.png": "08_match_scheduling_weekdays",
    "weekday_share_trends_by_season.png": "08_match_scheduling_weekdays",
    "weekday_total_match_count.png": "08_match_scheduling_weekdays",
    "weekday_weekly_load_boxplot.png": "08_match_scheduling_weekdays",
    # demographics
    "demographics_attendance_per_1000_by_season.png": "09_demographics_and_club_context",
    "demographics_budget_vs_attendance.png": "09_demographics_and_club_context",
    "demographics_model_r_squared_comparison.png": "09_demographics_and_club_context",
    "demographics_over_under_performers.png": "09_demographics_and_club_context",
    "demographics_population_vs_attendance.png": "09_demographics_and_club_context",
    "demographics_top_clubs_attendance_per_1000.png": "09_demographics_and_club_context",
    # economic
    "budget_analysis_overview.png": "10_economic_data_and_squad_values",
    "squad_values_overview.png": "10_economic_data_and_squad_values",
    "FOREIGN_PLAYERS_01_trends_over_time.png": "10_economic_data_and_squad_values",
    "FOREIGN_PLAYERS_02_nationalities_clubs.png": "10_economic_data_and_squad_values",
    "FOREIGN_PLAYERS_03_distributions_statistics.png": "10_economic_data_and_squad_values",
    "foreign_players_analysis.png": "10_economic_data_and_squad_values",
    # misc / league dynamics
    "stability_correlations.png": "11_league_dynamics_and_misc",
    "comparison_old_vs_new_scraping_method.png": "11_league_dynamics_and_misc",
    # title race — reports/figures (title_race + regular season analysis)
    "distinct_leaders_per_season.png": "01_title_race",
    "contenders_distribution.png": "01_title_race",
    "contenders_over_time_sample_seasons.png": "01_title_race",
    "contenders_per_season_bar.png": "01_title_race",
    "gap_boxplots_by_season.png": "01_title_race",
    "gap_over_time_sample_seasons.png": "01_title_race",
    "lead_timeline_multiple_seasons.png": "01_title_race",
    "leadership_change_percentage.png": "01_title_race",
    "leadership_changes_comprehensive.png": "01_title_race",
    "leadership_changes_per_season.png": "01_title_race",
    "leadership_changes_per_season_regular.png": "01_title_race",
    "leadership_heatmap.png": "01_title_race",
    "leadership_stability_score.png": "01_title_race",
    "points_gap_2010_11.png": "01_title_race",
    "points_gap_all_seasons_summary.png": "01_title_race",
    "points_gap_comparison.png": "01_title_race",
    "race_type_by_season_stacked.png": "01_title_race",
    "rounds_led_heatmap.png": "01_title_race",
    "seasons_avg_gap_comparison.png": "01_title_race",
    "competitiveness_analysis.png": "01_title_race",
}

# Source roots to scan (relative paths recorded in manifest)
SOURCE_ROOTS = [
    ROOT / "notebooks" / "outputs" / "research_summary" / "charts",
    ROOT / "notebooks" / "reports" / "figures",
    ROOT / "notebooks" / "data" / "processed" / "relegation_competitiveness",
    ROOT / "notebooks" / "data" / "processed" / "relegation_competitiveness" / "plots",
]

SOURCE_NOTEBOOKS: dict[str, str] = {
    "01_title_race": "title_race/, regular season/02_regular_season_analysis.ipynb, scripts/build_research_summary_table.py",
    "02_relegation_battles": "relegation_battles/, scripts/build_research_summary_table.py",
    "03_match_outcomes_and_goals": "regular season/, scripts/build_research_summary_table.py",
    "04_league_balance_and_structure": "regular season/02_regular_season_analysis.ipynb, league_dynamics/",
    "05_playoffs_vs_regular_season": "playoffs/, title_race/, scripts/build_research_summary_table.py",
    "06_attendance_overview": "attendance/01_attendance_collection.ipynb, attendance/02_attendance_analysis.ipynb",
    "07_attendance_by_match_day": "scripts/generate_attendance_day_charts.py, scripts/build_research_summary_table.py",
    "08_match_scheduling_weekdays": "regular season/03_weekday_scheduling_analysis.ipynb",
    "09_demographics_and_club_context": "demographic/demographics_analysis.ipynb",
    "10_economic_data_and_squad_values": "economic_data/economic_data_analysis.ipynb, economic_data/economic_data_collection.ipynb",
    "11_league_dynamics_and_misc": "league_dynamics/01_round_rank_changes.ipynb",
}


def classify(path: Path) -> str | None:
    name = path.name
    if name in EXPLICIT:
        return EXPLICIT[name]
    parent = path.parent.name
    if parent == "attendance_day_analysis":
        return "07_attendance_by_match_day"
    if parent == "weekday_analysis":
        return "08_match_scheduling_weekdays"
    return None


def relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_readme(manifest: list[tuple[str, str, str]]) -> None:
    lines = [
        "Ligat Ha'al — Research Figures by Topic",
        "=" * 42,
        "",
        "Curated copy of all PNG charts from notebooks and build scripts.",
        "Original files are unchanged; re-run this script after generating new charts.",
        "",
        "Suggested writing order",
        "-" * 22,
    ]
    for folder, desc in TOPICS.items():
        lines.append(f"  {folder}/")
        lines.append(f"    {desc}")
        lines.append(f"    Sources: {SOURCE_NOTEBOOKS.get(folder, 'see manifest.csv')}")
        lines.append("")

    lines.extend(["", "File counts per topic", "-" * 22])
    from collections import Counter

    counts = Counter(t for t, _, _ in manifest)
    for folder in TOPICS:
        lines.append(f"  {folder}: {counts.get(folder, 0)} charts")

    lines.extend(
        [
            "",
            "Manifest: manifest.csv lists every copied file with original path.",
            "Run: python scripts/organize_research_figures_by_topic.py",
        ]
    )
    (OUTPUT / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for folder in TOPICS:
        (OUTPUT / folder).mkdir(parents=True)

    manifest: list[tuple[str, str, str]] = []
    seen_dest: set[str] = set()

    png_by_rel: dict[str, Path] = {}
    for src_root in SOURCE_ROOTS:
        if not src_root.exists():
            continue
        for src in sorted(src_root.rglob("*.png")):
            rel = relative_source(src)
            png_by_rel.setdefault(rel, src)

    unmapped: list[str] = []
    for src in sorted(png_by_rel.values(), key=relative_source):
        topic = classify(src)
        if not topic:
            unmapped.append(relative_source(src))
            continue

        dest_name = src.name
        dest = OUTPUT / topic / dest_name
        if dest_name in seen_dest:
            stem, suffix = src.stem, src.suffix
            parent_hint = src.parent.name
            if parent_hint not in ("charts", "figures", "plots", "relegation_competitiveness"):
                dest_name = f"{parent_hint}__{stem}{suffix}"
            else:
                rel = relative_source(src).replace("/", "__")
                dest_name = rel.replace(".png", "") + ".png"
            dest = OUTPUT / topic / dest_name

        seen_dest.add(dest.name)
        shutil.copy2(src, dest)
        manifest.append((topic, dest.name, relative_source(src)))

    manifest.sort(key=lambda x: (x[0], x[1]))
    manifest_path = OUTPUT / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("topic_folder,filename,original_path\n")
        for topic, fname, orig in manifest:
            f.write(f"{topic},{fname},{orig}\n")

    write_readme(manifest)

    extras = [
        (
            ROOT
            / "notebooks"
            / "outputs"
            / "research_summary"
            / "charts"
            / "attendance_day_analysis"
            / "README_attendance_day_analysis.txt",
            OUTPUT / "07_attendance_by_match_day" / "README_attendance_day_analysis.txt",
        ),
    ]
    for src, dest in extras:
        if src.exists():
            shutil.copy2(src, dest)

    print("=" * 60)
    print("RESEARCH FIGURES ORGANIZED BY TOPIC")
    print("=" * 60)
    print(f"Output: {OUTPUT}")
    print(f"Total charts copied: {len(manifest)}")
    if unmapped:
        print(f"Skipped (unmapped): {len(unmapped)}")
        for u in unmapped:
            print(f"  - {u}")
    print()
    from collections import Counter

    for folder in TOPICS:
        n = sum(1 for t, _, _ in manifest if t == folder)
        print(f"  {folder}: {n} files")
    print()
    print("Index: README.txt, manifest.csv")


if __name__ == "__main__":
    main()
