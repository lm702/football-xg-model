import pandas as pd
import numpy as np
from scipy.stats import poisson

def compute_team_coefficients(df, window=10):
    """
    计算每支球队的主/客场攻防系数。
    window: 最近比赛场次（取最近N场用于计算均值）
    """
    # 分离主客队数据
    home_df = df[['date', 'home_team', 'home_npxg', 'away_npxg']].copy()
    home_df.rename(columns={'home_team': 'team', 'home_npxg': 'npxg_for', 'away_npxg': 'npxg_against'}, inplace=True)
    home_df['venue'] = 'home'
    
    away_df = df[['date', 'away_team', 'away_npxg', 'home_npxg']].copy()
    away_df.rename(columns={'away_team': 'team', 'away_npxg': 'npxg_for', 'home_npxg': 'npxg_against'}, inplace=True)
    away_df['venue'] = 'away'
    
    all_matches = pd.concat([home_df, away_df], ignore_index=True)
    all_matches.sort_values(['team', 'date'], inplace=True)
    
    # 取最近window场（按team分组）
    recent = all_matches.groupby('team').tail(window)
    
    # 按venue分别计算均值
    agg = recent.groupby(['team', 'venue']).agg(
        avg_npxg_for=('npxg_for', 'mean'),
        avg_npxg_against=('npxg_against', 'mean')
    ).unstack(level='venue')
    
    # flatten columns
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.rename(columns={
        'avg_npxg_for_home': 'ATT_H',
        'avg_npxg_for_away': 'ATT_A',
        'avg_npxg_against_home': 'DEF_H',
        'avg_npxg_against_away': 'DEF_A'
    })
    
    # 计算联赛均值 (基于同一window)
    # 注意: 联赛均值是所有主队/客队在该window中的平均值
    home_avg_xg = recent[recent['venue']=='home']['npxg_for'].mean()
    away_avg_xg = recent[recent['venue']=='away']['npxg_for'].mean()
    home_avg_def = recent[recent['venue']=='home']['npxg_against'].mean()
    away_avg_def = recent[recent['venue']=='away']['npxg_against'].mean()
    
    league_avg = {
        'Lg_ATT_H': home_avg_xg,
        'Lg_ATT_A': away_avg_xg,
        'Lg_DEF_H': home_avg_def,
        'Lg_DEF_A': away_avg_def
    }
    
    # 系数
    agg['CoA_H'] = agg['ATT_H'] / league_avg['Lg_ATT_H']
    agg['CoA_A'] = agg['ATT_A'] / league_avg['Lg_ATT_A']
    agg['CoD_H'] = agg['DEF_H'] / league_avg['Lg_DEF_H']   # >1 就是防守差
    agg['CoD_A'] = agg['DEF_A'] / league_avg['Lg_DEF_A']
    
    # 填充缺失值（比如某队没打过主场/客场）
    agg.fillna(1.0, inplace=True)
    
    return agg, league_avg

def compute_xpts(df, team_coeffs, league_avg):
    """
    用基于各场比赛npxg的泊松模拟，计算每队的预期积分(xPTS)和实际积分。
    返回DataFrame: team, actual_pts, xPTS, residual
    """
    # 计算每场比赛预期胜平负概率
    results = []
    for idx, row in df.iterrows():
        home_xg = row['home_npxg']
        away_xg = row['away_npxg']
        # 泊松模型
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
        
        results.append({
            'team': row['home_team'],
            'actual_pts': 3 if row['home_goals'] > row['away_goals'] else (1 if row['home_goals'] == row['away_goals'] else 0),
            'xPTS': home_xpts
        })
        results.append({
            'team': row['away_team'],
            'actual_pts': 3 if row['away_goals'] > row['home_goals'] else (1 if row['away_goals'] == row['home_goals'] else 0),
            'xPTS': away_xpts
        })
    
    pts_df = pd.DataFrame(results)
    summary = pts_df.groupby('team').sum()
    summary['residual'] = summary['actual_pts'] - summary['xPTS']
    return summary[['actual_pts', 'xPTS', 'residual']].sort_values('residual', ascending=False)