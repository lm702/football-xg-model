import numpy as np
import pandas as pd
from scipy.stats import poisson, weibull_min

# ---------- 基础时间强度曲线 ----------
def time_intensity_multiplier(current_minute, total_minutes=90):
    if current_minute <= 15:
        return 0.8
    elif current_minute <= 30:
        return 1.0
    elif current_minute <= 45:
        return 1.2
    elif current_minute <= 60:
        return 0.9
    elif current_minute <= 75:
        return 1.1
    else:
        return 1.3

def calculate_remaining_expected_goals(home_expg_full, away_expg_full,
                                       current_minute, current_home_goals,
                                       current_away_goals,
                                       home_attack_adj=1.0, away_attack_adj=1.0,
                                       total_minutes=90, injury_time=0):
    if current_minute >= 90:
        remaining_minutes = max(injury_time - (current_minute - 90), 0)
        time_mult_home = 1.5
        time_mult_away = 1.5
    else:
        remaining_minutes = 90 - current_minute + (injury_time if current_minute >= 85 else 0)
        mid = (current_minute + 90) / 2
        t1 = time_intensity_multiplier(current_minute)
        t2 = time_intensity_multiplier(mid)
        t3 = time_intensity_multiplier(90)
        time_mult_home = np.mean([t1, t2, t3])
        time_mult_away = time_mult_home

    uniform_h = home_expg_full * remaining_minutes / 90.0
    uniform_a = away_expg_full * remaining_minutes / 90.0
    remaining_h = uniform_h * time_mult_home
    remaining_a = uniform_a * time_mult_away

    goal_diff = current_home_goals - current_away_goals
    if goal_diff > 0:
        remaining_h *= 0.75
        remaining_a *= 1.2
    elif goal_diff < 0:
        remaining_h *= 1.2
        remaining_a *= 0.75

    remaining_h *= home_attack_adj
    remaining_a *= away_attack_adj
    remaining_h = max(remaining_h, 0.0)
    remaining_a = max(remaining_a, 0.0)
    return remaining_h, remaining_a

def inplay_probabilities(home_expg, away_expg, current_minute,
                         current_home_goals, current_away_goals,
                         home_attack_adj=1.0, away_attack_adj=1.0,
                         total_minutes=90, injury_time=0):
    lam_h, lam_a = calculate_remaining_expected_goals(
        home_expg, away_expg, current_minute, current_home_goals, current_away_goals,
        home_attack_adj, away_attack_adj, total_minutes, injury_time
    )
    max_goals = min(8, int(np.ceil(lam_h + lam_a + 2 * np.sqrt(lam_h + lam_a) + 2)))
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if current_home_goals + i > current_away_goals + j:
                home_win += p
            elif current_home_goals + i == current_away_goals + j:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    return home_win, draw, away_win, lam_h, lam_a

# ---------- 场面压制指数 ----------
def compute_pressure_index(df):
    home = df[['home_team', 'home_xg_open', 'home_shots', 'home_box_touches', 'home_possession']].copy()
    home.columns = ['team', 'xg_open', 'shots', 'box_touches', 'possession']
    home['venue'] = 'home'
    away = df[['away_team', 'away_xg_open', 'away_shots', 'away_box_touches', 'away_possession']].copy()
    away.columns = ['team', 'xg_open', 'shots', 'box_touches', 'possession']
    away['venue'] = 'away'
    all_df = pd.concat([home, away], ignore_index=True)
    all_df = all_df[all_df['shots'] > 0]
    all_df['penetration'] = all_df['xg_open'] / all_df['shots']
    agg = all_df.groupby('team').agg(
        penetration=('penetration', 'mean'),
        avg_shots=('shots', 'mean'),
        avg_box_touches=('box_touches', 'mean'),
        avg_possession=('possession', 'mean')
    )
    agg['pressure'] = agg['avg_box_touches'] / (agg['avg_possession'] + 0.01)
    return agg[['penetration', 'pressure']]

# ---------- 球队节奏标签 ----------
def compute_rhythm_labels(df):
    home = df[['home_team', 'home_shots', 'home_xg_set', 'home_npxg']].copy()
    home.columns = ['team', 'shots', 'xg_set', 'npxg']
    away = df[['away_team', 'away_shots', 'away_xg_set', 'away_npxg']].copy()
    away.columns = ['team', 'shots', 'xg_set', 'npxg']
    all_df = pd.concat([home, away], ignore_index=True)
    agg = all_df.groupby('team').agg(
        avg_shots=('shots', 'mean'),
        sum_xg_set=('xg_set', 'sum'),
        sum_npxg=('npxg', 'sum')
    )
    agg['set_ratio'] = agg['sum_xg_set'] / agg['sum_npxg'].replace(0, np.nan)
    agg['set_ratio'] = agg['set_ratio'].fillna(0)
    def label(row):
        if row['avg_shots'] > 14 or row['set_ratio'] < 0.25:
            return '高节奏'
        elif row['avg_shots'] < 10 or row['set_ratio'] > 0.35:
            return '低节奏'
        else:
            return '中节奏'
    return agg.apply(label, axis=1)

# ---------- 反向盘口概率 ----------
def reverse_implied_probs(odds_list):
    inv = [1/o for o in odds_list]
    total = sum(inv)
    return [i/total for i in inv]

# ---------- 红牌影响估算 ----------
def red_card_impact(home_expg, away_expg, minute, is_home, team_def_coeff, total_minutes=90):
    remaining = max(0, total_minutes - minute) / total_minutes
    attack_penalty = 0.6
    defense_penalty = 0.85
    if is_home:
        home_expg_new = home_expg * (1 - remaining) + home_expg * remaining * attack_penalty
        away_expg_new = away_expg * (1 - remaining) + away_expg * remaining * (1 / team_def_coeff * defense_penalty)
    else:
        away_expg_new = away_expg * (1 - remaining) + away_expg * remaining * attack_penalty
        home_expg_new = home_expg * (1 - remaining) + home_expg * remaining * (1 / team_def_coeff * defense_penalty)
    return max(home_expg_new, 0.0), max(away_expg_new, 0.0)

# ---------- 进球时间分布（Weibull）----------
def goal_time_probability(current_minute, remaining_minutes, c=2.2, scale=45):
    dist = weibull_min(c, loc=0, scale=scale)
    t1 = current_minute
    t2 = current_minute + remaining_minutes
    p_before = dist.cdf(t1)
    p_after = dist.cdf(t2)
    return p_after - p_before

# ---------- 球队进球后节奏变化（简化：平均射门）----------
def goal_rhythm_stats(df, team):
    home = df[df['home_team']==team]['home_shots'].mean()
    away = df[df['away_team']==team]['away_shots'].mean()
    return (home + away) / 2
