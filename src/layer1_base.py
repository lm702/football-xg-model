import pandas as pd
import numpy as np
from scipy.stats import poisson

def compute_team_coefficients(df, window=10, min_games=3):
    """
    计算每支球队的主/客场攻防系数，基于最近 window 场主场/客场各自独立计算。
    """
    # 1. 组建每支球队每场比赛的视角数据
    home_df = df[['date', 'home_team', 'home_npxg', 'away_npxg']].copy()
    home_df.rename(columns={'home_team': 'team', 'home_npxg': 'npxg_for', 'away_npxg': 'npxg_against'}, inplace=True)
    home_df['venue'] = 'home'

    away_df = df[['date', 'away_team', 'away_npxg', 'home_npxg']].copy()
    away_df.rename(columns={'away_team': 'team', 'away_npxg': 'npxg_for', 'home_npxg': 'npxg_against'}, inplace=True)
    away_df['venue'] = 'away'

    all_matches = pd.concat([home_df, away_df], ignore_index=True)
    all_matches.sort_values(['team', 'date'], inplace=True)

    # 2. 分 venue 取最近 window 场
    def venue_recent(group):
        """按 venue 分组后，取每个 venue 的最近 window 场"""
        result = []
        for venue, g in group.groupby('venue'):
            recent = g.sort_values('date').tail(window)
            result.append(recent)
        return pd.concat(result) if result else pd.DataFrame()

    recent_by_venue = all_matches.groupby('team', group_keys=False).apply(venue_recent)

    # 3. 聚合每个 (team, venue) 的均值
    agg = recent_by_venue.groupby(['team', 'venue']).agg(
        avg_npxg_for=('npxg_for', 'mean'),
        avg_npxg_against=('npxg_against', 'mean'),
        count=('npxg_for', 'count')
    ).reset_index()

    # 4. 构建宽表
    def pivot_venue(sub, venue):
        sub = sub[sub['venue'] == venue].set_index('team')
        return sub[['avg_npxg_for', 'avg_npxg_against', 'count']].rename(
            columns={'avg_npxg_for': f'ATT_{venue[0].upper()}', 
                     'avg_npxg_against': f'DEF_{venue[0].upper()}',
                     'count': f'cnt_{venue[0].upper()}'}
        )

    home_data = pivot_venue(agg, 'home')
    away_data = pivot_venue(agg, 'away')
    coeffs = home_data.join(away_data, how='outer')

    # 5. 对不足 min_games 的设为 NaN（后续用联赛均值填充或单独处理）
    for col in ['cnt_H', 'cnt_A']:
        if col in coeffs.columns:
            mask = coeffs[col] < min_games
            venue_letter = col[-1]  # H or A
            coeffs.loc[mask, [f'ATT_{venue_letter}', f'DEF_{venue_letter}']] = np.nan

    # 6. 计算联赛均值（只使用有足够样本的球队）
    valid_home = coeffs.dropna(subset=['ATT_H'])
    valid_away = coeffs.dropna(subset=['ATT_A'])

    league_avg = {
        'Lg_ATT_H': valid_home['ATT_H'].mean(),
        'Lg_ATT_A': valid_away['ATT_A'].mean(),
        'Lg_DEF_H': valid_home['DEF_H'].mean(),
        'Lg_DEF_A': valid_away['DEF_A'].mean()
    }

    # 7. 计算系数，缺失值用 1.0 填充（表示联赛平均）
    for venue in ['H', 'A']:
        att_col = f'ATT_{venue}'
        def_col = f'DEF_{venue}'
        if att_col in coeffs.columns and def_col in coeffs.columns:
            coeffs[f'CoA_{venue}'] = coeffs[att_col] / league_avg[f'Lg_ATT_{venue}']
            coeffs[f'CoD_{venue}'] = coeffs[def_col] / league_avg[f'Lg_DEF_{venue}']
    
    # 缺失的系数用 1.0 填充
    coeffs.fillna(1.0, inplace=True)

    # 清除辅助计数列
    coeffs = coeffs[[c for c in coeffs.columns if not c.startswith('cnt_')]]

    return coeffs, league_avg


def compute_xpts(df, team_coeffs, league_avg):
    """
    基于已完成的每场比赛的 npxg 计算 xPTS 与实际积分残差
    """
    results = []
    for idx, row in df.iterrows():
        home_xg = row['home_npxg']
        away_xg = row['away_npxg']
        max_goals = 8
        home_win_prob = 0
        draw_prob = 0
        for i in range(0, max_goals+1):
            for j in range(0, max_goals+1):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
        away_win_prob = 1 - home_win_prob - draw_prob

        home_xpts = 3 * home_win_prob + 1 * draw_prob
        away_xpts = 3 * away_win_prob + 1 * draw_prob

        results.append({'team': row['home_team'], 'actual_pts': 3 if row['home_goals'] > row['away_goals'] else (1 if row['home_goals'] == row['away_goals'] else 0), 'xPTS': home_xpts})
        results.append({'team': row['away_team'], 'actual_pts': 3 if row['away_goals'] > row['home_goals'] else (1 if row['away_goals'] == row['home_goals'] else 0), 'xPTS': away_xpts})

    pts_df = pd.DataFrame(results)
    summary = pts_df.groupby('team').sum()
    summary['residual'] = summary['actual_pts'] - summary['xPTS']
    return summary[['actual_pts', 'xPTS', 'residual']].sort_values('residual', ascending=False)
