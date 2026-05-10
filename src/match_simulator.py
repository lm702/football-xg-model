import numpy as np
import random

def simulate_match_timeline(home_expg, away_expg, match_length=90):
    """
    基于泊松过程模拟比赛进球时间线
    返回: 事件列表 [(分钟, 球队), ...]
    """
    np.random.seed()  # 确保每次调用随机
    events = []
    total_expg = home_expg + away_expg
    if total_expg <= 0:
        return events

    # 模拟总进球数（泊松）
    total_goals = np.random.poisson(total_expg)
    if total_goals == 0:
        return events

    # 生成进球时间（均匀分布）
    goal_times = sorted([random.randint(1, match_length) for _ in range(total_goals)])

    # 分配进球给主客队（按预期进球比例）
    for t in goal_times:
        if np.random.random() < home_expg / total_expg:
            events.append((t, 'home'))
        else:
            events.append((t, 'away'))

    # 按时间排序
    events.sort(key=lambda x: x[0])
    return events

def format_timeline(events, home_team, away_team):
    """将事件列表格式化为可读文本"""
    if not events:
        return f"{home_team} 0 - 0 {away_team} (无进球)"

    home_score = 0
    away_score = 0
    lines = []
    for minute, team in events:
        if team == 'home':
            home_score += 1
            lines.append(f"⚽ {minute}' {home_team} 进球！ ({home_score}-{away_score})")
        else:
            away_score += 1
            lines.append(f"⚽ {minute}' {away_team} 进球！ ({home_score}-{away_score})")

    timeline_text = "\n".join(lines)
    result = f"{home_team} {home_score} - {away_score} {away_team}\n\n{timeline_text}"
    return result
