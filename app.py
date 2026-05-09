import streamlit as st
import pandas as pd
from src.data_utils import load_and_clean
from src.layer1_base import compute_team_coefficients, compute_xpts
from src.layer2_trend import compute_trends
from src.layer3_scenario import (compute_setpiece_attack_ratio,
                                 compute_setpiece_defense_ratio,
                                 compute_shooting_quality)
from src.layer4_match import calibrate_expg, poisson_probabilities
from src.visual import plot_xg_trend, plot_match_matrix

st.set_page_config(page_title="足球xG推理模型", layout="wide")
st.title("⚽ 足球比赛深度推理模型（基于xG数据）")

uploaded_file = st.sidebar.file_uploader("上传联赛xlsx数据", type=["xlsx"])

if uploaded_file is not None:
    # 读取与清洗
    df = load_and_clean(uploaded_file)
    st.sidebar.success(f"成功加载 {len(df)} 场比赛数据")

    # 计算各项指标（缓存）
    @st.cache_data
    def compute_all(df):
        team_coeffs, league_avg = compute_team_coefficients(df, window=10)
        xpts_summary = compute_xpts(df, team_coeffs, league_avg)
        trends = compute_trends(df)
        set_att = compute_setpiece_attack_ratio(df)
        set_def = compute_setpiece_defense_ratio(df)
        shooting = compute_shooting_quality(df)
        return team_coeffs, league_avg, xpts_summary, trends, set_att, set_def, shooting

    team_coeffs, league_avg, xpts_summary, trends, set_att, set_def, shooting = compute_all(df)

    # 安全获取所有球队名（避免混合类型报错）
    all_teams = df['home_team'].dropna().astype(str).tolist() + df['away_team'].dropna().astype(str).tolist()
    teams = sorted(set(all_teams))

    # 侧边栏选择比赛
    st.sidebar.subheader("比赛推演")
    home_team = st.sidebar.selectbox("主队", teams, index=0)
    away_team = st.sidebar.selectbox("客队", teams, index=1)

    if home_team == away_team:
        st.sidebar.error("主客队不能相同")
    else:
        # 推演分析
        home_expg, away_expg = calibrate_expg(
            home_team, away_team, team_coeffs, league_avg, trends,
            set_att, set_def
        )

        # 泊松概率
        h_win, draw, a_win, prob_matrix = poisson_probabilities(home_expg, away_expg, dixon_coles_adjust=True)

        # 主界面展示
        col1, col2, col3 = st.columns(3)
        col1.metric("主队校准预期进球 (ExpG)", f"{home_expg:.2f}")
        col2.metric("客队校准预期进球 (ExpG)", f"{away_expg:.2f}")
        with col3:
            st.write("**胜平负概率**")
            st.write(f"主胜: {h_win:.2%}")
            st.write(f"平局: {draw:.2%}")
            st.write(f"客胜: {a_win:.2%}")

        # 比分矩阵
        st.plotly_chart(plot_match_matrix(prob_matrix, home_team, away_team), use_container_width=True)

        # 可选：市场赔率对比
        st.subheader("与市场赔率对比（可选）")
        with st.expander("输入赔率"):
            col_o1, col_o2, col_o3 = st.columns(3)
            mkt_h = col_o1.number_input("主胜赔率", value=0.0, step=0.01)
            mkt_d = col_o2.number_input("平局赔率", value=0.0, step=0.01)
            mkt_a = col_o3.number_input("客胜赔率", value=0.0, step=0.01)
            if mkt_h > 1.0 and mkt_d > 1.0 and mkt_a > 1.0:
                odds = [mkt_h, mkt_d, mkt_a]
                imp_probs = [(1/o) for o in odds]
                total = sum(imp_probs)
                imp_probs = [p/total for p in imp_probs]
                diff = [h_win - imp_probs[0], draw - imp_probs[1], a_win - imp_probs[2]]
                st.write("差值 (模型-市场):", dict(zip(['主胜','平局','客胜'], [f"{d:.2%}" for d in diff])))
                if diff[0] > 0.05:
                    st.info("模型认为主胜被低估")
                elif diff[2] > 0.05:
                    st.info("模型认为客胜被低估")

        # 详细信息：球队实力、趋势等
        st.header("球队深层数据")
        tab1, tab2, tab3 = st.tabs(["实力系数", "趋势 & 运气", "定位球与射门"])

        with tab1:
            st.dataframe(team_coeffs.style.format("{:.2f}"))

        with tab2:
            st.subheader("预期积分残差 (运气指标)")
            st.dataframe(xpts_summary.style.format("{:.2f}").bar(subset=['residual'], color=['#d65f5f' if x < 0 else '#5fba7d' for x in xpts_summary['residual']]))
            if home_team in trends:
                t = trends[home_team]
                st.write(f"{home_team} 近期xG净胜值变化: {t['delta_net']:.2f}")
            if away_team in trends:
                t = trends[away_team]
                st.write(f"{away_team} 近期xG净胜值变化: {t['delta_net']:.2f}")

        with tab3:
            st.subheader("定位球依赖")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("进攻定位球占比")
                st.dataframe(set_att.to_frame().style.format("{:.2%}"))
            with col_b:
                st.write("防守定位球占比（被创造xG中）")
                st.dataframe(set_def.to_frame().style.format("{:.2%}"))
            st.subheader("射门质量 (xGOT/射正)")
            st.dataframe(shooting.style.format("{:.2f}"))

        # 可选趋势图
        st.header("球队xG趋势")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.plotly_chart(plot_xg_trend(home_team, df), use_container_width=True)
        with col_t2:
            st.plotly_chart(plot_xg_trend(away_team, df), use_container_width=True)

else:
    st.info("👈 请上传一个符合格式的xlsx文件开始分析")
