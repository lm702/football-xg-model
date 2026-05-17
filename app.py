import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from src.data_utils import load_and_clean
from src.layer1_base import compute_team_coefficients, compute_xpts
from src.layer2_trend import compute_trends, compute_stability
from src.layer3_scenario import (compute_setpiece_attack_ratio,
                                 compute_setpiece_defense_ratio,
                                 compute_shooting_quality)
from src.layer4_match import calibrate_expg, poisson_probabilities, apply_match_context_adjustments
from src.visual import plot_xg_trend, plot_match_matrix
from src.match_simulator import simulate_match_timeline, format_timeline
from src.inplay import (inplay_probabilities, compute_pressure_index,
                        compute_rhythm_labels, reverse_implied_probs,
                        red_card_impact, goal_time_probability,
                        goal_rhythm_stats)
from src.deepseek_client import deepseek_chat

st.set_page_config(page_title="足球xG推理模型", layout="wide")
st.title("⚽ 足球比赛深度推理模型（基于xG数据）")

uploaded_file = st.sidebar.file_uploader("上传联赛xlsx数据", type=["xlsx"])

if uploaded_file is not None:
    df = load_and_clean(uploaded_file)
    st.sidebar.success(f"成功加载 {len(df)} 场比赛数据")

    @st.cache_data
    def compute_all(df):
        team_coeffs, league_avg = compute_team_coefficients(df, window=10)
        xpts_summary = compute_xpts(df, team_coeffs, league_avg)
        trends = compute_trends(df)
        set_att = compute_setpiece_attack_ratio(df)
        set_def = compute_setpiece_defense_ratio(df)
        shooting = compute_shooting_quality(df)
        stability = compute_stability(df, window=10)
        pressure_index = compute_pressure_index(df)
        rhythm_labels = compute_rhythm_labels(df)
        return (team_coeffs, league_avg, xpts_summary, trends,
                set_att, set_def, shooting, stability,
                pressure_index, rhythm_labels)

    (team_coeffs, league_avg, xpts_summary, trends,
     set_att, set_def, shooting, stability,
     pressure_index, rhythm_labels) = compute_all(df)

    all_teams = (df['home_team'].dropna().astype(str).tolist() +
                 df['away_team'].dropna().astype(str).tolist())
    teams = sorted(set(all_teams))

    mode = st.sidebar.radio("选择模式", ["实时预测", "历史回测", "走地分析"])

        # ========== 实时预测模式 ==========
    if mode == "实时预测":
        st.sidebar.subheader("比赛推演")
        home_team = st.sidebar.selectbox("主队", teams, index=0)
        away_team = st.sidebar.selectbox("客队", teams, index=1)

        if home_team == away_team:
            st.sidebar.error("主客队不能相同")
        else:
            # --- 临场修正面板（修正版，提前初始化动机变量）---
            st.sidebar.subheader("🧠 临场修正")
            apply_correction = st.sidebar.checkbox(
                "开启情景修正", value=False,
                help="根据战意、伤病、疲劳等软信息调整预期进球"
            )

            home_adj = 1.0
            away_adj = 1.0
            adj_log = []

            # 初始化软信息变量
            home_motivation = '正常'
            away_motivation = '正常'
            home_derby = False
            away_derby = False
            home_fatigue = False
            away_fatigue = False

            if apply_correction:
                st.sidebar.markdown("**战意锚点**")
                home_motivation = st.sidebar.selectbox(
                    f"{home_team} 比赛动机",
                    ['正常', '争冠关键战', '欧战资格关键战', '保级生死战', '无欲无求'],
                    key='home_motivation'
                )
                away_motivation = st.sidebar.selectbox(
                    f"{away_team} 比赛动机",
                    ['正常', '争冠关键战', '欧战资格关键战', '保级生死战', '无欲无求'],
                    key='away_motivation'
                )

                home_derby = st.sidebar.checkbox(f"{home_team} 是德比/宿敌战", key='home_derby')
                away_derby = st.sidebar.checkbox(f"{away_team} 是德比/宿敌战", key='away_derby')

                motivation_factors = {
                    '正常': 1.00,
                    '争冠关键战': 1.05,
                    '欧战资格关键战': 1.02,
                    '保级生死战': 0.95,
                    '无欲无求': 0.90
                }

                h_mot_mult = motivation_factors[home_motivation]
                a_mot_mult = motivation_factors[away_motivation]
                home_adj *= h_mot_mult
                away_adj *= a_mot_mult
                if home_motivation != '正常':
                    adj_log.append(f"{home_team} 因 '{home_motivation}' 修正 x{h_mot_mult:.2f}")
                if away_motivation != '正常':
                    adj_log.append(f"{away_team} 因 '{away_motivation}' 修正 x{a_mot_mult:.2f}")

                if home_derby:
                    home_adj *= 0.95
                    adj_log.append(f"{home_team} 德比战额外下调")
                if away_derby:
                    away_adj *= 0.95
                    adj_log.append(f"{away_team} 德比战额外下调")

                st.sidebar.markdown("**赛程疲劳**")
                home_fatigue = st.sidebar.checkbox(f"{home_team} 受赛程疲劳影响", key='home_fatigue')
                away_fatigue = st.sidebar.checkbox(f"{away_team} 受赛程疲劳影响", key='away_fatigue')
                if home_fatigue or away_fatigue:
                    fatigue_factor = st.sidebar.slider(
                        "疲劳削弱系数", 0.80, 1.00, 0.95, 0.01, key='fatigue_factor'
                    )
                    if home_fatigue:
                        home_adj *= fatigue_factor
                        adj_log.append(f"{home_team} 疲劳修正 x{fatigue_factor:.2f}")
                    if away_fatigue:
                        away_adj *= fatigue_factor
                        adj_log.append(f"{away_team} 疲劳修正 x{fatigue_factor:.2f}")

                st.sidebar.write(f"当前主队系数: {home_adj:.2f}")
                st.sidebar.write(f"当前客队系数: {away_adj:.2f}")
                manual_override = st.sidebar.checkbox("手动覆盖调整系数", key='manual_override')
                if manual_override:
                    home_adj = st.sidebar.slider("主队调整系数", 0.80, 1.20, float(home_adj), 0.01, key='home_adj_override')
                    away_adj = st.sidebar.slider("客队调整系数", 0.80, 1.20, float(away_adj), 0.01, key='away_adj_override')

                if adj_log:
                    st.sidebar.info("情景调整:\n" + "\n".join(adj_log))
            # --- 修正面板结束 ---

            home_expg, away_expg = calibrate_expg(
                home_team, away_team, team_coeffs, league_avg, trends,
                set_att, set_def
            )
            home_expg, away_expg = apply_match_context_adjustments(
                home_expg, away_expg, home_adj, away_adj
            )

            h_win, draw, a_win, prob_matrix = poisson_probabilities(
                home_expg, away_expg, dixon_coles_adjust=True
            )

            over25 = sum(p for (i,j), p in prob_matrix.items() if i+j > 2)
            btts = sum(p for (i,j), p in prob_matrix.items() if i>0 and j>0)
            home_clean = sum(p for (i,j), p in prob_matrix.items() if i>0 and j==0)
            away_clean = sum(p for (i,j), p in prob_matrix.items() if i==0 and j>0)

            col1, col2, col3 = st.columns(3)
            col1.metric("主队校准预期进球 (ExpG)", f"{home_expg:.2f}")
            col2.metric("客队校准预期进球 (ExpG)", f"{away_expg:.2f}")
            with col3:
                st.write("**胜平负概率**")
                st.write(f"主胜: {h_win:.2%}")
                st.write(f"平局: {draw:.2%}")
                st.write(f"客胜: {a_win:.2%}")

            st.subheader("🎲 常用玩法概率")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("大于2.5球", f"{over25:.1%}")
            col_p2.metric("双方进球 (BTTS)", f"{btts:.1%}")
            col_p3.metric("主队零封", f"{home_clean:.1%}")
            col_p4.metric("客队零封", f"{away_clean:.1%}")

            st.plotly_chart(
                plot_match_matrix(prob_matrix, home_team, away_team),
                use_container_width=True
            )

            st.subheader("与市场赔率对比（可选）")
            with st.expander("输入赔率"):
                col_o1, col_o2, col_o3 = st.columns(3)
                mkt_h = col_o1.number_input("主胜赔率", value=0.0, step=0.01)
                mkt_d = col_o2.number_input("平局赔率", value=0.0, step=0.01)
                mkt_a = col_o3.number_input("客胜赔率", value=0.0, step=0.01)
                if mkt_h > 1.0 and mkt_d > 1.0 and mkt_a > 1.0:
                    odds = [mkt_h, mkt_d, mkt_a]
                    imp_probs = [1/o for o in odds]
                    total = sum(imp_probs)
                    imp_probs = [p/total for p in imp_probs]
                    diff = [h_win - imp_probs[0], draw - imp_probs[1], a_win - imp_probs[2]]
                    st.write("差值 (模型-市场):", dict(zip(['主胜','平局','客胜'], [f"{d:.2%}" for d in diff])))
                    ev_h = h_win * mkt_h - 1
                    ev_d = draw * mkt_d - 1
                    ev_a = a_win * mkt_a - 1
                    if ev_h > 0.05:
                        st.success(f"🔥 主胜价值投注 (EV: +{ev_h:.2%})")
                    if ev_d > 0.05:
                        st.success(f"🔥 平局价值投注 (EV: +{ev_d:.2%})")
                    if ev_a > 0.05:
                        st.success(f"🔥 客胜价值投注 (EV: +{ev_a:.2%})")

            st.subheader("🎮 比赛模拟回放")
            if st.button("模拟一次比赛进程"):
                events = simulate_match_timeline(home_expg, away_expg)
                timeline = format_timeline(events, home_team, away_team)
                st.text(timeline)

            st.header("球队深层数据")
            tab1, tab2, tab3 = st.tabs(["实力系数", "趋势 & 运气", "定位球与射门"])

            with tab1:
                st.dataframe(team_coeffs.style.format("{:.2f}"))

            with tab2:
                st.subheader("预期积分残差 (运气指标)")
                st.dataframe(
                    xpts_summary.style.format("{:.2f}")
                    .bar(subset=['residual'], color=['#d65f5f', '#5fba7d'])
                )
                if home_team in trends:
                    t = trends[home_team]
                    st.write(f"{home_team} 近期xG净胜值变化: {t['delta_net']:.2f}")
                if away_team in trends:
                    t = trends[away_team]
                    st.write(f"{away_team} 近期xG净胜值变化: {t['delta_net']:.2f}")

                st.markdown("---")
                st.subheader("近10场稳定性评估 (xG净胜值标准差)")
                if home_team in stability:
                    s = stability[home_team]
                    st.write(f"{home_team}: σ={s['sigma']:.2f}，评级：**{s['rating']}**")
                else:
                    st.write(f"{home_team}: 数据不足")
                if away_team in stability:
                    s = stability[away_team]
                    st.write(f"{away_team}: σ={s['sigma']:.2f}，评级：**{s['rating']}**")
                else:
                    st.write(f"{away_team}: 数据不足")

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

            st.header("球队xG趋势")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.plotly_chart(plot_xg_trend(home_team, df), use_container_width=True)
            with col_t2:
                st.plotly_chart(plot_xg_trend(away_team, df), use_container_width=True)

            # ========== 冷门倾向分析 ==========
            st.markdown("---")
            st.subheader("❄️ 冷门倾向分析")
            with st.expander("点击查看冷门推演结果"):
                upset_score, upset_max, upset_reasons = compute_upset_score(
                    home_team, away_team,
                    home_expg, away_expg,
                    team_coeffs, xpts_summary, trends, shooting,
                    set_att, set_def, stability, rhythm_labels,
                    home_adj=home_adj, away_adj=away_adj,
                    home_motivation=home_motivation,
                    away_motivation=away_motivation,
                    home_fatigue=home_fatigue,
                    away_fatigue=away_fatigue,
                    home_derby=home_derby,
                    away_derby=away_derby
                )

                upset_pct = upset_score / upset_max
                color = 'green' if upset_pct < 0.4 else ('orange' if upset_pct < 0.7 else 'red')
                st.markdown(f"**冷门倾向评分：{upset_score}/{upset_max}**")
                st.progress(upset_pct, text=f"{upset_score}/{upset_max}")
                for reason in upset_reasons:
                    st.write(f"- {reason}")
                if upset_score >= 5:
                    st.warning("⚠️ 冷门风险较高，建议谨慎对待热门方向")

            # ---------- AI 深度解读 ----------
            st.markdown("---")
            st.subheader("🧠 DeepSeek 深度解读")
            if st.button("生成 AI 解读"):
                # 准备上下文数据
                trend_h_text = ""
                if home_team in trends:
                    t = trends[home_team]
                    trend_h_text = f"xG净胜值变化：{t['delta_net']:.2f}"
                trend_a_text = ""
                if away_team in trends:
                    t = trends[away_team]
                    trend_a_text = f"xG净胜值变化：{t['delta_net']:.2f}"

                stability_h_text = "无"
                if home_team in stability:
                    s = stability[home_team]
                    stability_h_text = f"σ={s['sigma']:.2f} ({s['rating']})"
                stability_a_text = "无"
                if away_team in stability:
                    s = stability[away_team]
                    stability_a_text = f"σ={s['sigma']:.2f} ({s['rating']})"

                set_h_val = set_att.get(home_team, 0) if hasattr(set_att, 'get') else 0
                set_def_a_val = set_def.get(away_team, 0) if hasattr(set_def, 'get') else 0
                set_h_text = f"{set_h_val:.0%}" if isinstance(set_h_val, (int, float)) else "无"
                set_def_a_text = f"{set_def_a_val:.0%}" if isinstance(set_def_a_val, (int, float)) else "无"

                prompt = f"""
你是一位资深足球分析师。以下是一场即将到来的比赛的数据分析摘要：

主队：{home_team}，客队：{away_team}
模型预期进球：{home_team} {home_expg:.2f} vs {away_team} {away_expg:.2f}
胜平负概率：主胜 {h_win:.1%}，平局 {draw:.1%}，客胜 {a_win:.1%}
大于2.5球概率：{over25:.1%}

双方近期趋势：
{home_team} {trend_h_text}，稳定性：{stability_h_text}
{away_team} {trend_a_text}，稳定性：{stability_a_text}

定位球依赖：
{home_team} 进攻定位球占比 {set_h_text}
{away_team} 防守定位球占比 {set_def_a_text}

临场调整系数：主队 x{home_adj:.2f}，客队 x{away_adj:.2f}
调整说明：{'; '.join(adj_log) if adj_log else '无'}

冷门倾向评分：{upset_score}/{upset_max}，主要因素：{'; '.join(upset_reasons)}

请用简洁专业的语言：
1. 分析模型概率背后的关键驱动因素。
2. 指出本场比赛最值得关注的战术点（如节奏、定位球、防守稳定性）。
3. 结合冷门信号，提供1-2条滚球观察建议（基于剩余时间、比分等，但暂未进行中）。
注意：你的分析不构成投注建议，仅供决策参考。
"""
                with st.spinner("AI 正在思考..."):
                    analysis = deepseek_chat(prompt)
                if analysis:
                    st.success("AI 解读生成完毕")
                    st.markdown(analysis)
                else:
                    st.warning("无法获取 AI 解读，请检查 API Key 或网络。")

    # ========== 历史回测模式 ==========
    elif mode == "历史回测":
        st.sidebar.subheader("📊 回测追踪")
        st.sidebar.markdown("选择一场已发生的比赛，对比模型预测与实际结果")

        if 'backtest_records' not in st.session_state:
            st.session_state['backtest_records'] = []

        df_valid = df.dropna(subset=['date']).sort_values('date', ascending=False)
        if df_valid.empty:
            st.warning("没有可用日期进行回测")
            st.stop()

        match_options = [
            f"{row['date'].strftime('%Y-%m-%d')}  {row['home_team']} vs {row['away_team']}"
            for _, row in df_valid.iterrows()
        ]
        selected_match_str = st.sidebar.selectbox("选择比赛", match_options)
        selected_idx = match_options.index(selected_match_str)
        match_row = df_valid.iloc[selected_idx]

        col_info1, col_info2 = st.columns(2)
        col_info1.write(f"**日期**: {match_row['date'].strftime('%Y-%m-%d')}")
        col_info1.write(f"**轮次**: {match_row['round']}")
        col_info2.write(f"**主队**: {match_row['home_team']}")
        col_info2.write(f"**客队**: {match_row['away_team']}")

        actual_home_goals = match_row['home_goals']
        actual_away_goals = match_row['away_goals']
        col_info1.metric("实际比分", f"{actual_home_goals} - {actual_away_goals}")

        if st.sidebar.button("运行模型预测"):
            home_expg, away_expg = calibrate_expg(
                match_row['home_team'], match_row['away_team'],
                team_coeffs, league_avg, trends,
                set_att, set_def
            )
            h_win, draw, a_win, prob_matrix = poisson_probabilities(
                home_expg, away_expg, dixon_coles_adjust=True
            )

            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("主胜概率", f"{h_win:.1%}")
            col_p2.metric("平局概率", f"{draw:.1%}")
            col_p3.metric("客胜概率", f"{a_win:.1%}")
            st.write(f"模型预期进球: 主 {home_expg:.2f} - 客 {away_expg:.2f}")

            def outcome(actual_h, actual_a):
                if actual_h > actual_a:
                    return 'H'
                elif actual_h == actual_a:
                    return 'D'
                else:
                    return 'A'

            actual_out = outcome(actual_home_goals, actual_away_goals)
            pred_probs = {'H': h_win, 'D': draw, 'A': a_win}
            predicted_out = max(pred_probs, key=pred_probs.get)
            is_correct = (predicted_out == actual_out)

            brier = (h_win - (1 if actual_out == 'H' else 0))**2 + \
                    (draw - (1 if actual_out == 'D' else 0))**2 + \
                    (a_win - (1 if actual_out == 'A' else 0))**2
            brier /= 3

            col_acc1, col_acc2 = st.columns(2)
            col_acc1.metric("方向正确?", "✅" if is_correct else "❌")
            col_acc2.metric("Brier分数 (越低越好)", f"{brier:.3f}")

            if st.button("📌 保存这次回测记录"):
                record = {
                    '日期': match_row['date'],
                    '主队': match_row['home_team'],
                    '客队': match_row['away_team'],
                    '实际比分': f"{actual_home_goals}-{actual_away_goals}",
                    '预测方向': '主胜' if predicted_out == 'H' else ('平局' if predicted_out == 'D' else '客胜'),
                    '实际结果': '主胜' if actual_out == 'H' else ('平局' if actual_out == 'D' else '客胜'),
                    '是否正确': is_correct,
                    'Brier分数': round(brier, 3)
                }
                st.session_state['backtest_records'].append(record)
                st.success("已保存！")

        if st.session_state['backtest_records']:
            st.subheader("📋 回测记录历史")
            records_df = pd.DataFrame(st.session_state['backtest_records'])
            st.dataframe(records_df)

            total = len(records_df)
            correct = records_df['是否正确'].sum()
            accuracy = correct / total if total > 0 else 0
            avg_brier = records_df['Brier分数'].mean() if total > 0 else 0
            col_stats1, col_stats2 = st.columns(2)
            col_stats1.metric("方向准确率", f"{accuracy:.1%} ({correct}/{total})")
            col_stats2.metric("平均Brier分数", f"{avg_brier:.3f}")

            if st.button("🗑️ 清空回测记录"):
                st.session_state['backtest_records'] = []
                st.rerun()

    # ========== 走地分析模式 ==========
    else:
        st.sidebar.subheader("🔄 资深滚球分析")
        home_team = st.sidebar.selectbox("主队", teams, index=0)
        away_team = st.sidebar.selectbox("客队", teams, index=1)
        if home_team == away_team:
            st.sidebar.error("主客队不能相同")
        else:
            home_expg_pre, away_expg_pre = calibrate_expg(
                home_team, away_team, team_coeffs, league_avg, trends, set_att, set_def
            )
            st.sidebar.write(f"赛前模型ExpG：{home_team} {home_expg_pre:.2f} - {away_team} {away_expg_pre:.2f}")
            use_custom = st.sidebar.checkbox("手动调整赛前ExpG")
            if use_custom:
                home_expg_pre = st.sidebar.slider("主队全场ExpG", 0.0, 6.0, float(home_expg_pre), 0.05)
                away_expg_pre = st.sidebar.slider("客队全场ExpG", 0.0, 6.0, float(away_expg_pre), 0.05)

            st.sidebar.markdown("---")
            st.sidebar.markdown("**当前比赛状态**")
            current_min = st.sidebar.slider("当前分钟数", 0, 120, 45, 1)
            cur_h_goals = st.sidebar.number_input(f"{home_team} 进球", 0, 20, 0)
            cur_a_goals = st.sidebar.number_input(f"{away_team} 进球", 0, 20, 0)
            injury = st.sidebar.slider("预计伤停补时（分钟）", 0, 15, 3, 1)

            st.sidebar.markdown("**手动态势调整**")
            home_momentum = st.sidebar.slider(f"{home_team} 攻击态势", 0.5, 1.5, 1.0, 0.05, help=">1进攻更猛")
            away_momentum = st.sidebar.slider(f"{away_team} 攻击态势", 0.5, 1.5, 1.0, 0.05)

            st.sidebar.markdown("**突发事件**")
            red_card_team = st.sidebar.radio("红牌方", ["无", home_team, away_team])
            red_minute = st.sidebar.number_input("红牌发生分钟（若选择）", 0, 120, 0, 1)

            # 自动计算走地概率（无按钮）
            home_adj_expg = home_expg_pre
            away_adj_expg = away_expg_pre
            if red_card_team != "无" and red_minute > 0 and red_minute <= current_min:
                is_home = (red_card_team == home_team)
                if is_home:
                    def_coeff = team_coeffs.loc[home_team, 'CoD_H'] if home_team in team_coeffs.index else 1.0
                else:
                    def_coeff = team_coeffs.loc[away_team, 'CoD_A'] if away_team in team_coeffs.index else 1.0
                home_adj_expg, away_adj_expg = red_card_impact(
                    home_expg_pre, away_expg_pre, red_minute, is_home, def_coeff
                )

            hw, dr, aw, lam_h, lam_a = inplay_probabilities(
                home_adj_expg, away_adj_expg, current_min, cur_h_goals, cur_a_goals,
                home_momentum, away_momentum, injury_time=injury
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("主胜概率", f"{hw:.1%}")
            col2.metric("平局概率", f"{dr:.1%}")
            col3.metric("客胜概率", f"{aw:.1%}")
            st.write(f"剩余预期进球：{home_team} {lam_h:.2f} - {away_team} {lam_a:.2f}")

            # 模块1：盘口背离
            with st.expander("📉 场面与盘口背离检测"):
                pre_line = st.number_input("赛前让球", value=0.0, step=0.25)
                live_line = st.number_input("实时让球", value=0.0, step=0.25)
                if pre_line != 0 or live_line != 0:
                    theoretical_line = (home_expg_pre - away_expg_pre) * 0.8
                    st.write(f"模型理论让球：{theoretical_line:.2f}")
                    if abs(theoretical_line - live_line) > 0.3:
                        st.warning("⚠️ 盘口与预期存在明显背离")

            # 模块2：节奏预判
            with st.expander("🎵 节奏预判"):
                rhythm_h = rhythm_labels.get(home_team, '中节奏')
                rhythm_a = rhythm_labels.get(away_team, '中节奏')
                st.write(f"{home_team}: {rhythm_h}   |   {away_team}: {rhythm_a}")
                if cur_h_goals + cur_a_goals > 0:
                    st.info("已发生进球，关注节奏变化。")

            # 模块3：反向盘口
            with st.expander("🔁 下一个进球盘口反推"):
                odds_h = st.number_input("主队进球赔率", 1.0, 10.0, 2.20, 0.05)
                odds_d = st.number_input("无进球/平赔", 1.0, 10.0, 3.50, 0.05)
                odds_a = st.number_input("客队进球赔率", 1.0, 10.0, 2.80, 0.05)
                if odds_h > 1.0 and odds_a > 1.0 and odds_d > 1.0:
                    imp = reverse_implied_probs([odds_h, odds_d, odds_a])
                    st.write(f"市场隐含概率：主进 {imp[0]:.1%} | 无球 {imp[1]:.1%} | 客进 {imp[2]:.1%}")
                    model_next = 1 - poisson.cdf(0, lam_h + lam_a)
                    st.write(f"模型下一球概率：{model_next:.1%}")
                    if lam_h + lam_a > 0:
                        model_h = model_next * lam_h / (lam_h + lam_a)
                        model_a = model_next * lam_a / (lam_h + lam_a)
                        st.write(f"模型拆解：主进 {model_h:.1%} | 客进 {model_a:.1%}")
                        if model_h > imp[0] * 1.1:
                            st.success("主队进球被低估？")
                        if model_a > imp[2] * 1.1:
                            st.success("客队进球被低估？")

            # 模块4：时间概率
            with st.expander("⏳ 进球时间概率"):
                remaining = 90 + injury - current_min
                if remaining > 0:
                    p_goal = goal_time_probability(current_min, remaining)
                    st.write(f"剩余 {remaining} 分钟内至少一球的概率（Weibull）：{p_goal:.1%}")

            # 模块5：压制指数
            with st.expander("📊 场面压制力参考"):
                try:
                    pen_h = pressure_index.loc[home_team, 'penetration']
                    pen_a = pressure_index.loc[away_team, 'penetration']
                    pre_h = pressure_index.loc[home_team, 'pressure']
                    pre_a = pressure_index.loc[away_team, 'pressure']
                    st.write(f"{home_team}：穿透力 {pen_h:.2f}  压迫效率 {pre_h:.1f}")
                    st.write(f"{away_team}：穿透力 {pen_a:.2f}  压迫效率 {pre_a:.1f}")
                except:
                    st.write("数据不足")

            # ---------- AI 滚球顾问 ----------
            st.markdown("---")
            st.subheader("🧠 AI 滚球顾问")
            if st.button("生成滚球建议"):
                rhythm_h = rhythm_labels.get(home_team, '中节奏')
                rhythm_a = rhythm_labels.get(away_team, '中节奏')
                set_h_val = set_att.get(home_team, 0) if hasattr(set_att, 'get') else 0
                set_def_a_val = set_def.get(away_team, 0) if hasattr(set_def, 'get') else 0
                set_h_text = f"{set_h_val:.0%}" if isinstance(set_h_val, (int, float)) else "无"
                set_def_a_text = f"{set_def_a_val:.0%}" if isinstance(set_def_a_val, (int, float)) else "无"

                pen_h_val = pen_a_val = pre_h_val = pre_a_val = "无"
                try:
                    pen_h_val = f"{pressure_index.loc[home_team, 'penetration']:.2f}"
                    pen_a_val = f"{pressure_index.loc[away_team, 'penetration']:.2f}"
                    pre_h_val = f"{pressure_index.loc[home_team, 'pressure']:.1f}"
                    pre_a_val = f"{pressure_index.loc[away_team, 'pressure']:.1f}"
                except:
                    pass

                prompt = f"""
你是一位滚球交易分析师。当前比赛进行中，数据如下：

主队：{home_team}，客队：{away_team}
当前比分：{cur_h_goals} - {cur_a_goals}，当前时间：第{current_min}分钟（补时{injury}分钟）
剩余预期进球：{home_team} {lam_h:.2f} vs {away_team} {lam_a:.2f}
实时胜平负概率：主胜 {hw:.1%}，平局 {dr:.1%}，客胜 {aw:.1%}

球队风格：
{home_team}：节奏 {rhythm_h}，穿透力 {pen_h_val}，压迫 {pre_h_val}
{away_team}：节奏 {rhythm_a}，穿透力 {pen_a_val}，压迫 {pre_a_val}

定位球：
{home_team} 进攻定位球占比 {set_h_text}，{away_team} 防守定位球占比 {set_def_a_text}

手动态势调整：{home_team} x{home_momentum:.2f}，{away_team} x{away_momentum:.2f}
红牌情况：{red_card_team}（{red_minute}分钟）

请基于以上信息：
1. 评估当前比赛态势，指出哪一方更可能控制剩余时间。
2. 指出是否存在“场面与盘口背离”的可能。
3. 提供1-2条具体关注点（如下一个进球方向、大小球等）。
保持简洁，注重逻辑。
"""
                with st.spinner("AI 正在分析..."):
                    analysis = deepseek_chat(prompt)
                if analysis:
                    st.success("AI 建议生成完毕")
                    st.markdown(analysis)
                else:
                    st.warning("无法获取 AI 建议，请检查 API Key 或网络。")

else:
    st.info("👈 请上传一个符合格式的xlsx文件开始分析")
