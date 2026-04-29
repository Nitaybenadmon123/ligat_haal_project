m1_rows = []
warn_msgs = []

for p in sorted(REG_FINAL_DIR.glob('regular_final_table_*.csv')):
    season = season_label(p.name)
    if not season:
        continue
    N = len(pd.read_csv(p))
    if N not in (12, 14, 16):
        continue
    mpath = match_file_for_season(season)
    if mpath is None:
        m1_rows.append({
            'season': season,
            'checkpoint_round': np.nan,
            'total_regular_rounds': np.nan,
            'n_teams_at_checkpoint': np.nan,
            'survival_points': np.nan,
            'min_points_among_contenders': np.nan,
            'max_points_among_contenders': np.nan,
            'n_relegation_contenders': np.nan,
            'contender_teams': None,
            'survival_gap_max_pts': METRIC1_SURVIVAL_BAND_PTS,
            'note': 'No match file',
        })
        continue
    m = load_matches(mpath)
    rmax = int(m['round'].max())
    chk = checkpoint_round(rmax, 0.75)
    st = standings_after_round(m, chk).reset_index(drop=True)
    st['rank_snap'] = np.arange(1, len(st) + 1)

    sep = '=' * 72
    print(sep + '\n[M1] Season ' + str(season) + ' | N = ' + str(N) + ' | checkpoint round = ' + str(chk) + ' / ' + str(rmax))

    if len(st) != N:
        warn = (
            f'Season {season}: Checkpoint standings have {len(st)} teams '
            f'but regular final lists {N}; skip Metric 1 for this season.'
        )
        print(warn)
        warn_msgs.append(warn)
        m1_rows.append({
            'season': season,
            'checkpoint_round': chk,
            'total_regular_rounds': rmax,
            'n_teams_at_checkpoint': len(st),
            'survival_points': np.nan,
            'min_points_among_contenders': np.nan,
            'max_points_among_contenders': np.nan,
            'n_relegation_contenders': np.nan,
            'contender_teams': None,
            'survival_gap_max_pts': METRIC1_SURVIVAL_BAND_PTS,
            'note': warn,
        })
        continue

    lowest_safe_idx = N - N_REL - 1
    lowest_safe_points = int(st.iloc[lowest_safe_idx]['points'])

    st_ev = st.copy()
    st_ev['gap_to_survival'] = (st_ev['points'] - lowest_safe_points).abs()
    cand = st_ev[st_ev['gap_to_survival'] <= METRIC1_SURVIVAL_BAND_PTS].copy()

    rank_floor = N - 6
    cont = cand[cand['rank_snap'] >= rank_floor].copy()

    if len(cont) and (cont['rank_snap'] < rank_floor).any():
        msg = (
            'ERROR: contender(s) with rank_snap < '
            + str(rank_floor)
            + ' (must not appear above cutoff)'
        )
        print(msg)
        warn_msgs.append(season + ': ' + msg)

    print(f'lowest_safe_points (lowest_safe_idx={lowest_safe_idx}): {lowest_safe_points}')
    print(
        f'Band: abs(points - lowest_safe_points) <= {METRIC1_SURVIVAL_BAND_PTS}; '
        f'rank_snap >= N - 6 = {rank_floor}'
    )

    excluded_by_rank = cand[cand['rank_snap'] < rank_floor]
    if len(excluded_by_rank):
        print(
            'Within band but excluded (rank above bottom block '
            + f'(rank_snap < {rank_floor})):'
        )
        print(
            excluded_by_rank[
                ['rank_snap', 'team', 'points', 'gap_to_survival']
            ].to_string(index=False)
        )

    bot6 = st.tail(6)[['rank_snap', 'team', 'points']].copy()
    print('Full bottom 6 teams at checkpoint:')
    print(bot6.to_string(index=False))

    mn_c = float(cont['points'].min()) if len(cont) else float('nan')
    mx_c = float(cont['points'].max()) if len(cont) else float('nan')

    note_parts = ['Regular-season snapshot from matches']

    print('Contenders list:')
    if len(cont):
        print(cont[['rank_snap', 'team', 'points', 'gap_to_survival']].to_string(index=False))
        print(f'min contender pts = {mn_c}, max contender pts = {mx_c}')
    else:
        print('(none)')
        print('Warning: No relegation battle')
        note_parts.append('No relegation battle')

    if len(cont) == 1:
        print('Warning: only 1 contender — likely incorrect (verify snapshot)')
        note_parts.append('Only one contender — likely incorrect')

    m1_rows.append({
        'season': season,
        'checkpoint_round': chk,
        'total_regular_rounds': rmax,
        'n_teams_at_checkpoint': N,
        'survival_points': lowest_safe_points,
        'min_points_among_contenders': mn_c,
        'max_points_among_contenders': mx_c,
        'n_relegation_contenders': len(cont),
        'contender_teams': '; '.join(cont['team'].astype(str)) if len(cont) else '',
        'survival_gap_max_pts': METRIC1_SURVIVAL_BAND_PTS,
        'note': ' — '.join(note_parts),
    })
    print(sep)

if warn_msgs:
    print('\n'.join(warn_msgs))

metric1_df = pd.DataFrame(m1_rows).sort_values('season')
metric1_df.to_csv(OUT_DIR / 'metric1_relegation_contenders_index.csv', index=False, encoding='utf-8-sig')
display(metric1_df)
