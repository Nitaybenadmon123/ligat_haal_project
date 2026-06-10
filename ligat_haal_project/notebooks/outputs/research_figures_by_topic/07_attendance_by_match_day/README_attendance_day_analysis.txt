Attendance Day-of-Week Analysis
=============================

Data rules
----------
1. Valid attendance: Attendance > 0
2. Season inclusion: coverage >= 60% of matches in that season's combined regular+playoff files
3. Low-sample flag: weekdays with fewer than 30 valid matches are marked as low sample
4. Dates parsed from match files; weekday names in English (Monday-Sunday)
5. Regular season and playoff matches are combined per season

Included seasons (15): 2009/10, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26
Valid matches in analysis pool: 3396

Chart guide
-----------
01_attendance_boxplot_by_weekday.png
    Distribution, median, and outliers per weekday. Use to see whether Friday's high average is driven by outliers.

02_attendance_mean_vs_median_by_weekday.png
    Compares mean and median. Large gaps suggest outlier influence.

03_attendance_match_scatter_by_weekday.png
    Every match as a point with jitter; red markers/lines show weekday means. Shows Friday's tiny sample visually.

04_attendance_reliable_days_only.png
    Average attendance only for weekdays with n >= 30. Low-sample days excluded from ranking.

05_attendance_share_vs_match_share_by_weekday.png
    Compares share of matches vs share of total spectators. Efficiency ratio > 1 means overperformance per match volume.

06_saturday_vs_weekday_attendance_trend.png
    Season-by-season Saturday average vs non-Saturday average for included seasons.

07_attendance_mean_ci_by_weekday.png
    Mean attendance with 95% CI (1.96 * SE). Purple bars = low sample; Friday CI is wide/unreliable due to n=7.

08_top_attendance_matches_by_weekday.png
    Which weekdays host the top 10 single-match attendance records.

Important note on Friday
------------------------
Friday shows a very high average attendance in the full sample, but typically has only a handful of matches (low sample).
Treat Friday as a potential scheduling opportunity, not a confirmed league-wide trend, until more Friday games are observed.
