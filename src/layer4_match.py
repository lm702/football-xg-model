import numpy as np
import pandas as pd
from scipy.stats import poisson
from src.layer3_scenario import setpiece_mismatch_adjustment

def calibrate_expg(home_team, away_team, team_coeffs, league_avg, trends,
                   attack_ratios, defense_ratios):
    """
    计算校准后的预期进球。
    返回: (home_expg, away_expg)
    """
    # 获取系数
    if home_team in team_coeffs.index:
        coa_h = team_coeffs.loc[home_team, 'CoA_H']
        cod_a = team_coeffs.loc[away_team, 'CoD_A']  # 客队客场防守系数
    else:
        coa_h, cod_a = 1.0, 1.0

    if away_team in team_coeffs.index:
        coa_a = team_coeffs.loc[away_team, 'CoA_A']
        cod_h = team_coeffs.loc[home_team, 'CoD_H']  # 主队主场防守系数
    else:
        coa_a, cod_h = 1.0, 1.0

    # 联赛均值
    lg_att_h = league_avg['Lg_ATT_H']
    lg_att_a = league_avg['Lg_ATT_A']

    # 基础预期
    home_expg = lg_att_h * coa_h * cod_a
    away_expg = lg_att_a * coa_a * cod_h

    # 趋势调整
    home_delta_net = trends.get(home_team, {}).get('delta_net', 0) * 0.3
    away_delta_net = trends.get(away_team, {}).get('delta_net', 0) * 0.3
    home_expg *= (1 + home_delta_net)
    away_expg *= (1 + away_delta_net)

    # 定位球错配调整
    home_adj, away_adj = setpiece_mismatch_adjustment(
        home_team, away_team, attack_ratios, defense_ratios
    )
    home_expg *= home_adj
    away_expg *= away_adj

    # 不能小于0.1
    home_expg = max(home_expg, 0.1)
    away_expg = max(away_expg, 0.1)

    return home_expg, away_expg

def poisson_probabilities(lam_h, lam_a, max_goals=8, dixon_coles_adjust=False):
    """返回胜平负概率及比分矩阵"""
    probs = {}
    for i in range(0, max_goals+1):
        for j in range(0, max_goals+1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if dixon_coles_adjust and i <= 1 and j <= 1:
                # 简化Dixon-Coles调整（低分相关性）
                rho = -0.1
                if i == 0 and j == 0:
                    p *= (1 + rho)
                elif i == 1 and j == 0:
                    p *= (1 - rho * lam_h)
                elif i == 0 and j == 1:
                    p *= (1 - rho * lam_a)
                elif i == 1 and j == 1:
                    p *= (1 + rho * lam_h * lam_a)
            probs[(i, j)] = p

    # 归一化
    total = sum(probs.values())
    for key in probs:
        probs[key] /= total

    # 胜平负
    home_win = sum(p for (i, j), p in probs.items() if i > j)
    draw = sum(p for (i, j), p in probs.items() if i == j)
    away_win = 1 - home_win - draw

    return home_win, draw, away_win, probs

def apply_match_context_adjustments(home_expg, away_expg, home_adj, away_adj):
    """在计算概率前，手动调整预期进球"""
    return home_expg * home_adj, away_expg * away_adj
