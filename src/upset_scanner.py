import pandas as pd

def compute_upset_score(
    home_team, away_team,
    home_expg, away_expg,
    team_coeffs,
    xpts_summary,
    trends,
    shooting,
    set_att,
    set_def,
    stability,
    rhythm_labels,
    home_adj=1.0,
    away_adj=1.0,
    home_motivation='正常',
    away_motivation='正常',
    home_fatigue=False,
    away_fatigue=False,
    home_derby=False,
    away_derby=False
):
    """
    计算本场比赛的冷门倾向评分（0~10）。
    返回: (score, max_score, reasons_list)
    """
    reasons = []
    score = 0
    max_score = 10

    # 1. 确定强队和弱队
    expg_diff = home_expg - away_expg
    if abs(expg_diff) < 0.3:
        return 0, max_score, ["两队实力接近，冷门倾向不显著"]

    if expg_diff > 0:
        strong_team = home_team
        weak_team = away_team
        strong_side = 'home'
        weak_side = 'away'
        strong_expg = home_expg
        weak_expg = away_expg
        strong_motivation = home_motivation
        weak_motivation = away_motivation
        strong_fatigue = home_fatigue
        weak_fatigue = away_fatigue
        strong_derby = home_derby
        weak_derby = away_derby
    else:
        strong_team = away_team
        weak_team = home_team
        strong_side = 'away'
        weak_side = 'home'
        strong_expg = away_expg
        weak_expg = home_expg
        strong_motivation = away_motivation
        weak_motivation = home_motivation
        strong_fatigue = away_fatigue
        weak_fatigue = home_fatigue
        strong_derby = away_derby
        weak_derby = home_derby

    # 辅助函数：安全获取球队数据
    def get_coeff(team, var):
        try:
            return team_coeffs.loc[team, var]
        except (KeyError, AttributeError):
            return None

    # 2. xPTS残差（强队）
    try:
        strong_residual = xpts_summary.loc[strong_team, 'residual']
        if strong_residual > 6:
            score += 2
            reasons.append(f"{strong_team} 预期积分残差 {strong_residual:.1f} (>+6)，可能虚高")
    except (KeyError, AttributeError):
        pass

    # 3. 近期趋势（强队δ为负）
    if strong_team in trends:
        delta = trends[strong_team].get('delta_net', 0)
        if delta < -0.2:
            score += 1
            reasons.append(f"{strong_team} 近期xG净胜值下滑 (δ={delta:.2f})")

    # 4. 转化率过高（强队）
    if strong_team in shooting.index:
        conv = shooting.loc[strong_team, 'conversion']
        if pd.notna(conv) and conv > 1.2:
            score += 1
            reasons.append(f"{strong_team} 射门转化率异常高 ({conv:.2f})，可能回调")

    # 5. 弱队防守系数低（说明防守好）
    if weak_side == 'home':
        def_coeff = get_coeff(weak_team, 'CoD_H')
    else:
        def_coeff = get_coeff(weak_team, 'CoD_A')
    if def_coeff is not None and def_coeff < 0.8:
        score += 1
        reasons.append(f"{weak_team} 防守系数低 ({def_coeff:.2f})，防线稳固")

    # 6. 定位球错配
    try:
        weak_set_att = set_att.loc[weak_team] if weak_team in set_att.index else 0
        strong_set_def = set_def.loc[strong_team] if strong_team in set_def.index else 0
        if weak_set_att > 0.35 and strong_set_def > 0.35:
            score += 2
            reasons.append(f"定位球错配：{weak_team} 攻定位球 {weak_set_att:.0%}，{strong_team} 防定位球 {strong_set_def:.0%}")
    except:
        pass

    # 7. 风格克制（简单的节奏差异）
    rhythm_strong = rhythm_labels.get(strong_team, '中节奏')
    rhythm_weak = rhythm_labels.get(weak_team, '中节奏')
    if rhythm_strong == '高节奏' and rhythm_weak == '低节奏':
        score += 1
        reasons.append(f"{strong_team} 高节奏 vs {weak_team} 低节奏，防反可能奏效")

    # 8. 战意/疲劳/德比（软信息）
    # 强队无欲无求，弱队保级
    if strong_motivation in ['无欲无求', '正常'] and weak_motivation in ['保级生死战', '争冠关键战']:
        score += 1
        reasons.append(f"战意错位：{strong_team} 缺乏动力，{weak_team} 战意强烈")
    if strong_fatigue:
        score += 1
        reasons.append(f"{strong_team} 有赛程疲劳")
    if strong_derby or weak_derby:
        score += 1
        reasons.append("德比战，实力差距缩小")

    # 9. 强队稳定性
    if strong_team in stability:
        s = stability[strong_team]
        if s.get('rating') == '不稳定':
            score += 1
            reasons.append(f"{strong_team} 近期表现不稳定 (σ={s['sigma']:.2f})")

    # 分数上限
    score = min(score, max_score)

    if score == 0:
        reasons = ["暂无显著冷门信号"]

    return score, max_score, reasons
