Advanced Competitiveness Features
===================================

Offensive concentration
- top_scorer_goals: goals by league top scorer (Transfermarkt scorers list)
- share_of_league_goals_by_top_3_scorers: top 3 scorers' goals / season total goals * 100
- offensive_concentration_gini: Gini over all players' goals (higher = more concentrated)
Sources: notebooks/data/raw/player_stats/top_scorers_*_ligat_haal_transfermarkt.csv

Rank dynamics
- Parsed from notebooks/data/processed/rank_dynamics/rank_changes_export.csv
- Late-season window = last 5 rank transitions in season timeline
- Title zone = prev_rank <= 4; relegation zone = bottom 4 in cohort

Combined competitiveness
- Mean of min-max normalized title_race_closeness, relegation_closeness, league_balance (0-100)
- Requires at least 2 of 3 components

Playoff impact
- Championship: leadership changes and first-place flip during championship playoff only
- Relegation: zone membership changes and survival flips during relegation playoff only
- Descriptive only; not a statistical test

Data quality
- combined_data_quality_score: attendance (25) + goals (20) + assists (15) + playoff (20) + rank dynamics (20)

Known limitations
- Assists unavailable for 2007/08 and 2008/09 on Transfermarkt
- Attendance incomplete for seasons with coverage < 60%
- Budget coverage is partial (demographics join)
- Playoff impact is descriptive unless tested statistically
- Friday/weekday attendance charts are unchanged by this module
