import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_xg_trend(team, df):
    """球队近N场xG净胜趋势"""
    # 提取该队的比赛
    home = df[df['home_team']==team][['date', 'home_xg_net']].rename(columns={'home_xg_net':'xG_net'})
    away = df[df['away_team']==team][['date', 'away_xg_net']].rename(columns={'away_xg_net':'xG_net'})
    history = pd.concat([home, away]).sort_values('date')
    history['game_number'] = range(1, len(history)+1)
    
    fig = px.line(history, x='game_number', y='xG_net', title=f'{team} xG净胜值趋势',
                  labels={'xG_net':'xG净胜值', 'game_number':'比赛序号'})
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig

def plot_match_matrix(prob_matrix, home_team, away_team):
    """比分概率热力图"""
    max_g = 4
    data = []
    for i in range(max_g+1):
        row = []
        for j in range(max_g+1):
            row.append(prob_matrix.get((i, j), 0) * 100)
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[f'{j}' for j in range(max_g+1)],
        y=[f'{i}' for i in range(max_g+1)],
        colorscale='Blues',
        texttemplate='%{z:.1f}%',
        textfont={"size":10},
        hovertemplate='主队进球=%{y}<br>客队进球=%{x}<br>概率=%{z:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        title=f'{home_team} vs {away_team} 比分概率矩阵',
        xaxis_title=f'{away_team} 进球',
        yaxis_title=f'{home_team} 进球'
    )
    return fig