import pandas as pd
import numpy as np

def load_and_clean(uploaded_file):
    """读取xlsx并返回清洗后的DataFrame"""
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # 期望的原始列名（按你的描述）
    expected_cols = [
        '日期', '轮次', '主队', '主队比分', '客队比分', '客队',
        '主队xG', '客队xG', '主队xG Open Play', '客队xG Open Play',
        '主队xG Set Play', '客队xG Set Play', '主队Non-Pen xG', '客队Non-Pen xG',
        '主队xGOT', '客队xGOT', '主队控球率', '客队控球率',
        '主队射门', '客队射门', '主队射正', '客队射正',
        '主队对方禁区触球', '客队对方禁区触球'
    ]
    
    # 如果列名不完全匹配，尝试兼容
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"上传文件缺少必要列：{missing}")
    
    # 重命名为英文简写，方便后续处理
    rename_map = {
        '日期': 'date',
        '轮次': 'round',
        '主队': 'home_team',
        '主队比分': 'home_goals',
        '客队比分': 'away_goals',
        '客队': 'away_team',
        '主队xG': 'home_xg',
        '客队xG': 'away_xg',
        '主队xG Open Play': 'home_xg_open',
        '客队xG Open Play': 'away_xg_open',
        '主队xG Set Play': 'home_xg_set',
        '客队xG Set Play': 'away_xg_set',
        '主队Non-Pen xG': 'home_npxg',
        '客队Non-Pen xG': 'away_npxg',
        '主队xGOT': 'home_xgot',
        '客队xGOT': 'away_xgot',
        '主队控球率': 'home_possession',
        '客队控球率': 'away_possession',
        '主队射门': 'home_shots',
        '客队射门': 'away_shots',
        '主队射正': 'home_sot',
        '客队射正': 'away_sot',
        '主队对方禁区触球': 'home_box_touches',
        '客队对方禁区触球': 'away_box_touches'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 确保日期是datetime
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    # 衍生字段
    df['home_xg_net'] = df['home_npxg'] - df['away_npxg']
    df['away_xg_net'] = -df['home_xg_net']
    
    # 控球率可能是百分比字符串，转为浮点数
    for col in ['home_possession', 'away_possession']:
        if df[col].dtype == object:
            df[col] = df[col].str.rstrip('%').astype(float) / 100.0
    
    # 删除含有缺失值的行（关键列）
    key_cols = ['home_npxg', 'away_npxg', 'home_team', 'away_team']
    df.dropna(subset=key_cols, inplace=True)
    
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df