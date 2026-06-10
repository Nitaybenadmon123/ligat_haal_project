Ligat Ha'al — Research Figures by Topic
==========================================

Curated copy of all PNG charts from notebooks and build scripts.
Original files are unchanged; re-run this script after generating new charts.

Suggested writing order
----------------------
  01_title_race/
    Title race competitiveness, leadership changes, points gaps
    Sources: title_race/, regular season/02_regular_season_analysis.ipynb, scripts/build_research_summary_table.py

  02_relegation_battles/
    Relegation zone volatility, survival battles, closeness
    Sources: relegation_battles/, scripts/build_research_summary_table.py

  03_match_outcomes_and_goals/
    Home/draw/away rates and goals per match
    Sources: regular season/, scripts/build_research_summary_table.py

  04_league_balance_and_structure/
    League balance, Gini, structural comparisons
    Sources: regular season/02_regular_season_analysis.ipynb, league_dynamics/

  05_playoffs_vs_regular_season/
    Regular season vs playoff differences
    Sources: playoffs/, title_race/, scripts/build_research_summary_table.py

  06_attendance_overview/
    Season/team attendance levels, trends, forecasts
    Sources: attendance/01_attendance_collection.ipynb, attendance/02_attendance_analysis.ipynb

  07_attendance_by_match_day/
    Attendance by weekday (scheduling impact on crowds)
    Sources: scripts/generate_attendance_day_charts.py, scripts/build_research_summary_table.py

  08_match_scheduling_weekdays/
    When matches are played (volume, not attendance)
    Sources: regular season/03_weekday_scheduling_analysis.ipynb

  09_demographics_and_club_context/
    Population, budget, attendance efficiency by club
    Sources: demographic/demographics_analysis.ipynb

  10_economic_data_and_squad_values/
    Budgets, squad values, foreign players
    Sources: economic_data/economic_data_analysis.ipynb, economic_data/economic_data_collection.ipynb

  11_league_dynamics_and_misc/
    Rank changes, stability, data-quality checks
    Sources: league_dynamics/01_round_rank_changes.ipynb


File counts per topic
----------------------
  01_title_race: 29 charts
  02_relegation_battles: 9 charts
  03_match_outcomes_and_goals: 2 charts
  04_league_balance_and_structure: 6 charts
  05_playoffs_vs_regular_season: 7 charts
  06_attendance_overview: 16 charts
  07_attendance_by_match_day: 9 charts
  08_match_scheduling_weekdays: 9 charts
  09_demographics_and_club_context: 6 charts
  10_economic_data_and_squad_values: 6 charts
  11_league_dynamics_and_misc: 2 charts

Manifest: manifest.csv lists every copied file with original path.
Run: python scripts/organize_research_figures_by_topic.py