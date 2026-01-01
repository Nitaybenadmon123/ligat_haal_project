# Archive: Manual Calculation Files

**Date Archived**: January 1, 2026

## Purpose
This folder contains CSV files that were created by manual calculation methods and are no longer actively used in the current analysis workflow. They have been preserved here for historical reference and potential future needs.

## Why These Files Were Archived

The project transitioned from **manual calculations** to **Transfermarkt web scraping** for all data collection. The scraped data is more accurate and consistent, making these manually calculated files obsolete for active analysis.

## What Was Archived

### Old Interim Files (35 files)
Located in: `old_interim_files/`

1. **Standings Calculations** (26 files)
   - `standings_by_round_YYYY_YY.csv` - Manual standings calculations per season
   - `standings_by_round_YYYY_YY_final.csv` - Final standings for specific seasons
   - **Replaced by**: `data/interim/scraped_standings/all_seasons_gaps_scraped.csv`

2. **Title Race Analysis** (5 files)
   - `title_race_summary.csv`
   - `tm_title_race_per_round.csv`
   - `tm_title_race_summary.csv`
   - `tm_title_race_summary_aligned.csv`
   - `tm_title_race_summary_all_seasons.csv`
   - **Replaced by**: Files in `data/interim/scraped_standings/` and `data/processed/title_race_analysis/`

3. **Playoff Analysis** (2 files)
   - `playoff_championship_per_round.csv`
   - `playoff_championship_leadership_changes.csv`
   - **Replaced by**: `data/interim/scraped_standings/playoff_gaps_all_seasons.csv`

4. **Position Tracking** (5 files)
   - `positions_by_round_YYYY_YY.csv` - Position tracking for specific seasons
   - `positions_by_round_leader_changes_summary_tm.csv`
   - **Replaced by**: Scraped standings data

## Current Active Data Sources

All current analysis uses these files:

### Primary Scraped Data
- `data/interim/scraped_standings/all_seasons_gaps_scraped.csv` - Regular season gaps
- `data/interim/scraped_standings/playoff_gaps_all_seasons.csv` - Playoff gaps
- `data/matches/matches_*_ligat_haal_transfermarkt.csv` - Match results

### Processed Analysis
- `data/processed/title_race_analysis/` - All title race analysis outputs
- `data/interim/scraped_standings/regular_final_tables/` - Final standings
- `data/interim/scraped_standings/playoff_final_tables/` - Playoff final tables

## If You Need These Files

These archived files are still available and can be:
1. Referenced for historical comparison
2. Used to validate current scraping methods
3. Restored if needed for specific analysis

## Safe to Delete?

**No immediate deletion recommended.** Keep for at least one year to ensure all analysis transitions are complete. After validation that scraped data covers all use cases, these files can be safely deleted.

## File Counts

- **Total archived files**: 35+
- **Disk space saved in active folders**: Approximately 5-10 MB
- **Organization benefit**: Cleaner data structure, easier to find current data

---

**Note**: This archive was created automatically during the project cleanup to maintain only Transfermarkt scraped data in active analysis folders.
