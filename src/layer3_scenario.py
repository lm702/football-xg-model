import pandas as pd
import numpy as np

def setpiece_analysis(df):
    """计算每队的定位球攻防占比，基于赛季总数据的近似"""
    # 构建球队总数据
    home_df = df[['home_team', 'home_npxg', 'home_xg_set', 'away_xg_set']].copy()
    home_df.columns = ['team', 'npxg', 'xg_set_for', 'xg_set_against']
    home_df['venue'] = 'home'
    
    away_df = df[['away_team', 'away_npxg', 'away_xg_set', 'home_xg_set']].copy()
    away_df.columns = ['team', 'npxg', 'xg_set_for', 'xg_set_against']
    away_df['venue'] = 'away'
    
    all_data = pd.concat([home_df, away_df])
    
    totals = all_data.groupby('team').sum()
    # 定位球进攻占比 = sum(xg_set_for) / sum(npxg) ，npxg 不含点球，但已经包含set play xG
    # 注意：home_npxg 已包含 set play xG，所以可以用
    totals['set_attack_ratio'] = totals['xg_set_for'] / totals['npxg']
    totals['set_defense_ratio'] = totals['xg_set_against'] / totals['npxg']  # 注意这里分母是该队总npxg，代表该队让对手通过定位球创造的xG占自身总xG的比例，可能不太对。应该用对手对该队的总xG。但简单起见，我们用同一个表，可用'xg_set_against'除以该队被对手创造的总npxg。我们缺少"被对手总npxg"，可以换用 totals['npxg'] 作为该队总进攻量，不对。正确的防守占比：该队所有比赛被对手的set play xG之和 / 对手的总npxg之和。我们用原始df重新算。
    return totals[['set_attack_ratio', 'set_defense_ratio']]

def compute_setpiece_defense_ratio(df):
    """计算每队被对手通过定位球创造的xG占被对手总npxg的比重"""
    # 主队防守：对手（客队）的set play xG / 客队npxg
    home_def = df[['home_team', 'away_xg_set', 'away_npxg']].copy()
    home_def.columns = ['team', 'xg_set_allowed', 'total_npxg_allowed']
    away_def = df[['away_team', 'home_xg_set', 'home_npxg']].copy()
    away_def.columns = ['team', 'xg_set_allowed', 'total_npxg_allowed']
    
    defense = pd.concat([home_def, away_def])
    agg = defense.groupby('team').sum()
    agg['set_defense_ratio'] = agg['xg_set_allowed'] / agg['total_npxg_allowed']
    return agg['set_defense_ratio']

def compute_setpiece_attack_ratio(df):
    """定位球进攻占比：set play xG / 该队总npxg"""
    home_att = df[['home_team', 'home_xg_set', 'home_npxg']].copy()
    home_att.columns = ['team', 'xg_set', 'total_npxg']
    away_att = df[['away_team', 'away_xg_set', 'away_npxg']].copy()
    away_att.columns = ['team', 'xg_set', 'total_npxg']
    attack = pd.concat([home_att, away_att])
    agg = attack.groupby('team').sum()
    agg['set_attack_ratio'] = agg['xg_set'] / agg['total_npxg']
    return agg['set_attack_ratio']

def compute_shooting_quality(df):
    """射门质量指标：xGOT / 射正，以及转化率"""
    # 组合
    home = df[['home_team', 'home_xgot', 'home_sot', 'home_goals', 'home_npxg']].copy()
    home.columns = ['team', 'xgot', 'sot', 'goals', 'npxg']
    away = df[['away_team', 'away_xgot', 'away_sot', 'away_goals', 'away_npxg']].copy()
    away.columns = ['team', 'xgot', 'sot', 'goals', 'npxg']
    all_shots = pd.concat([home, away])
    agg = all_shots.groupby('team').sum()
    agg['xgot_per_sot'] = agg['xgot'] / agg['sot'].replace(0, np.nan)
    agg['conversion'] = agg['goals'] / agg['npxg'].replace(0, np.nan)
    return agg[['xgot_per_sot', 'conversion']].fillna(0)

def setpiece_mismatch_adjustment(home_team, away_team, attack_ratios, defense_ratios):
    """
    根据定位球错配返回调整系数。
    attack_ratios, defense_ratios: Series (index=team)
    返回: home_adj, away_adj (float multiplier)
    """
    default = 1.0
    if home_team not in attack_ratios or away_team not in defense_ratios:
        return default, default
    
    home_att_ratio = attack_ratios[home_team]
    away_def_ratio = defense_ratios[away_team]
    away_att_ratio = attack_ratios[away_team] if away_team in attack_ratios else 0.3
    home_def_ratio = defense_ratios[home_team]
    
    # 阈值
    high_att = 0.35
    low_def = 0.25  # low defense ratio means good at defending set pieces
    
    home_adj = 1.0
    away_adj = 1.0
    
    # 如果主队定位球进攻依赖度高且客队防定位球好（比率低），下调主队预期
    if home_att_ratio > high_att and away_def_ratio < low_def:
        home_adj = 0.92
    elif home_att_ratio > high_att and away_def_ratio > 0.35:  # 客队定位球防守差
        home_adj = 1.08
    
    if away_att_ratio > high_att and home_def_ratio < low_def:
        away_adj = 0.92
    elif away_att_ratio > high_att and home_def_ratio > 0.35:
        away_adj = 1.08
        
    return home_adj, away_adj