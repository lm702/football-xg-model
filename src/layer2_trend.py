import pandas as pd
import numpy as np

def compute_trends(df, short_window=5, long_window=10):
    """
    计算每支球队近short_window场的场均xG净胜值与长周期均值的差值δ。
    返回dict: {team: delta_net, delta_att, delta_def}
    """
    teams = pd.unique(df['home_team'].tolist() + df['away_team'].tolist())
    trends = {}
    for team in teams:
        # 提取该队所有比赛
        home = df[df['home_team']==team][['date', 'home_npxg', 'away_npxg']].rename(
            columns={'home_npxg':'npxg_for', 'away_npxg':'npxg_against'})
        home['venue'] = 'home'
        away = df[df['away_team']==team][['date', 'away_npxg', 'home_npxg']].rename(
            columns={'away_npxg':'npxg_for', 'home_npxg':'npxg_against'})
        away['venue'] = 'away'
        matches = pd.concat([home, away]).sort_values('date')
        if len(matches) < long_window:
            continue
        
        long_matches = matches.tail(long_window)
        short_matches = matches.tail(short_window)
        
        long_net = (long_matches['npxg_for'] - long_matches['npxg_against']).mean()
        short_net = (short_matches['npxg_for'] - short_matches['npxg_against']).mean()
        
        long_att = long_matches['npxg_for'].mean()
        short_att = short_matches['npxg_for'].mean()
        
        long_def = long_matches['npxg_against'].mean()
        short_def = short_matches['npxg_against'].mean()
        
        trends[team] = {
            'delta_net': short_net - long_net,
            'delta_att': short_att - long_att,
            'delta_def': short_def - long_def,  # 正数代表防守变差（对手xG变多）
            'short_net': short_net,
            'long_net': long_net
        }
    return trends