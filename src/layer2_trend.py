import pandas as pd

def compute_trends(df, short_window=5, long_window=10):
    """
    计算每支球队近short_window场的场均xG净胜值与长周期均值的差值δ。
    返回dict: {team: delta_net, delta_att, delta_def}
    """
    # 安全获取所有唯一球队名称——修复TypeError（混合数据类型）
    all_home = df['home_team'].dropna().astype(str).tolist()
    all_away = df['away_team'].dropna().astype(str).tolist()
    teams = sorted(set(all_home + all_away))

    trends = {}
    for team in teams:
        # 提取该队所有比赛
        home = df[df['home_team'] == team][['date', 'home_npxg', 'away_npxg']].rename(
            columns={'home_npxg': 'npxg_for', 'away_npxg': 'npxg_against'})
        home['venue'] = 'home'
        away = df[df['away_team'] == team][['date', 'away_npxg', 'home_npxg']].rename(
            columns={'away_npxg': 'npxg_for', 'home_npxg': 'npxg_against'})
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
            'delta_def': short_def - long_def,  # 正数表示防守变差
            'short_net': short_net,
            'long_net': long_net
        }
    return trends
def compute_stability(df, window=10):
    """
    计算每支球队近 window 场的 xG_Net 标准差，并返回带评级的 DataFrame
    """
    # 收集所有球队的主客场比赛的 xG_Net
    records = []
    for _, row in df.iterrows():
        records.append({
            'team': row['home_team'],
            'xg_net': row['home_npxg'] - row['away_npxg']
        })
        records.append({
            'team': row['away_team'],
            'xg_net': row['away_npxg'] - row['home_npxg']
        })
    all_records = pd.DataFrame(records)
    
    # 按球队取最近 window 场
    stability_stats = {}
    for team, group in all_records.groupby('team'):
        recent = group.tail(window)
        if len(recent) >= 3:  # 最少 3 场才计算
            sigma = recent['xg_net'].std()
            # 评级
            if sigma < 1.2:
                rating = '稳定'
            elif sigma < 1.8:
                rating = '中等'
            else:
                rating = '不稳定'
            stability_stats[team] = {'sigma': sigma, 'rating': rating}
        else:
            stability_stats[team] = {'sigma': None, 'rating': '数据不足'}
    
    return stability_stats
