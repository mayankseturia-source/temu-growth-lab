"""Temu Growth Lab - live analytics dashboard for the SBR case study (3 problems x 9 data-driven growth blogs).
All data is simulated. Run locally:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.ui import (AQUA, ARM_COLOR, BLUE, C, CRIT, GOOD, GREEN, MAGENTA, ORANGE, RED, SEG_COLOR, VIOLET, WARN,
                    YELLOW, blogs, fig_style, finding, kpi, money, pct)

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
st.set_page_config(page_title="Temu Growth Lab", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
ui.inject()


# ============================================================ loading
@st.cache_data(show_spinner=False)
def load():
    R = json.loads((ART / "results.json").read_text())
    A = {p.stem: pd.read_parquet(p) for p in ART.glob("*.parquet")}
    return R, A


@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load(ROOT / "models" / "uplift_models.joblib")


@st.cache_resource(show_spinner="Loading 30,000 simulated shoppers scored by the uplift models…")
def shopper_pool():
    f = ART / "live_pool.parquet"          # pre-scored by train/export step (fast, no model unpickling)
    if f.exists():
        return pd.read_parquet(f)
    from src.live import build_pool
    m = load_models()
    return build_pool(m["models"], m["columns"])


R, A = load()
P1, P2, P3 = R["problem1"], R["problem2"], R["problem3"]

# ============================================================ sidebar
with st.sidebar:
    st.markdown("### 🎛️ Live controls")
    live_on = st.toggle("Stream live data", value=True)
    speed = st.select_slider("Refresh every", options=[2, 3, 5, 10], value=3, format_func=lambda s: f"{s} s")
    traffic = st.slider("Shopper traffic", 0.5, 3.0, 1.0, 0.25)
    st.caption("Inject an incident and watch the live tab react:")
    inc = None
    if st.button("🛍️ Mega-sale traffic spike", use_container_width=True, help="x3 traffic for ~12 ticks"):
        inc = "spike"
    if st.button("🚚 Delivery delay storm", use_container_width=True, help="carrier backlog in GCC + EU, angry reviews"):
        inc = "delay"
    if st.button("📣 Notification blast", use_container_width=True, help="every user gets 6 more pushes a day: sleeping dogs wake up"):
        inc = "blast"
    if st.button("🔄 Restart simulation", use_container_width=True):
        st.session_state.pop("sim", None)
    st.divider()
    st.markdown("### 🧮 Problem 1 assumptions")
    st.caption("Uplift in 90-day repeat probability per action (edit to stress-test the projection)")
    e_high = st.slider("High value, high risk: loyalty reward + fast delivery", 0.0, 0.30, float(P1["effects"]["High value, high risk"]), 0.01)
    e_mid = st.slider("Mid value: offer timed to next purchase day", 0.0, 0.30, float(P1["effects"]["Mid value"]), 0.01)
    e_low = st.slider("Low value: free push message", 0.0, 0.10, float(P1["effects"]["Low value"]), 0.01)
    st.divider()
    d = R["data"]
    st.markdown("### 🗂️ Data (simulated)")
    st.caption(f"{d['customers']:,} customers · {d['orders']:,} orders ({d['order_period'][0]} → {d['order_period'][1]}) · "
               f"{d['coupon_experiment_users']:,} users in the coupon experiment · {d['warehouse_series']} warehouse×category "
               f"series × {d['weeks']} weeks · {d['skus']} SKUs · {d['reviews']:,} reviews")
    st.caption(f"Models trained {R['generated_at'][:16].replace('T', ' ')} UTC · seed {R['seed']}")

# ============================================================ header
st.markdown(f"""
<div class="hero">
  <span class="badge">🧪 SIMULATED DATA</span><span class="badge">🎓 SBR group project</span>
  <span class="badge">🤖 hyperparameter-tuned models</span>
  <h1>Temu Growth Lab</h1>
  <p>Three problems, nine data-driven growth techniques, one live dashboard. A student case study, not affiliated
  with Temu or PDD Holdings. Every number comes from simulated data built to match the group's Minutes of Meeting.</p>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🔴 Live", "🏆 Findings", "📏 Metrics", "🔁 P1 Retention", "🎟️ P2 Coupons",
                "📦 P3 Warehouses", "💬 Reviews", "🧪 Model lab", "📐 Formulas"])


# ============================================================ LIVE
def live_tab():
    from src.live import LiveSim
    if "sim" not in st.session_state:
        sim = LiveSim(shopper_pool(), guardrail=P2.get("guardrail_lambda_usd_per_order", 25.0),
                      fv=P2.get("future_value_usd", 6.4))
        sim.warm_up(40)
        st.session_state.sim = sim
    sim = st.session_state.sim
    if inc:
        sim.trigger(inc)

    @st.fragment(run_every=speed if live_on else None)
    def body():
        sim.step(traffic)
        T = sim.tick_df()
        V = sim.visits_df(60)
        last = T.tail(20)
        cum = T.copy()
        head = st.columns([3, 2])
        status = "LIVE" if live_on else "PAUSED"
        head[0].markdown(f'<span class="pulse"></span><b>{status}</b> · tick {sim.tick} · simulated clock '
                         f'{sim.now():%H:%M:%S} UTC · {len(V):,} shoppers in view'
                         + ("".join(f' · <span class="badge" style="background:#7a2a2a">{k} incident</span>' for k in sim.incident)),
                         unsafe_allow_html=True)
        head[1].markdown('<div class="small" style="text-align:right">Live A/B test: half of shoppers get the old '
                         'blanket 20% coupon, half get <b>smart targeting</b> (uplift model + cost optimisation)</div>',
                         unsafe_allow_html=True)

        su, bu = cum["smart_users"].sum(), cum["blanket_users"].sum()
        sv, bv = cum["smart_value"].sum() / su, cum["blanket_value"].sum() / bu
        sp = cum["smart_cost"].sum() / cum["smart_rev"].sum() * 100
        bp = cum["blanket_cost"].sum() / cum["blanket_rev"].sum() * 100
        so = cum["smart_orders"].sum() / su * 100
        bo = cum["blanket_orders"].sum() / bu * 100
        stockout = last["short"].sum() / max(last["units"].sum(), 1) * 100
        rv = sim.reviews.head(60)
        rv_ok = rv[~rv["spam_flag"]] if len(rv) else rv
        sent = rv_ok["sentiment"].mean() if len(rv_ok) else 0
        k = st.columns(6)
        kpi(k[0], "Shoppers · last minute", f"{int(last['visitors'].sum()):,}", f"{int(last['orders'].sum()):,} orders", "g1")
        kpi(k[1], "Revenue · last minute", money(last["revenue"].sum()), f"AOV {money(last['revenue'].sum() / max(last['orders'].sum(), 1), 2)}", "g2")
        kpi(k[2], "Promo cost % of sales", f"{sp:.1f}% vs {bp:.1f}%", "smart vs blanket (cumulative)", "g3")
        kpi(k[3], "Value per shopper", f"{money(sv, 2)} vs {money(bv, 2)}", f"smart wins by {money(sv - bv, 2)} incl. CLV", "g5")
        kpi(k[4], "Conversion", f"{so:.1f}% vs {bo:.1f}%", "smart vs blanket", "g7")
        kpi(k[5], "Stockouts · last minute", f"{stockout:.1f}%", f"review mood {sent:+.2f}", "g6" if stockout > 3 else "g4")

        a, b = st.columns(2)
        cv = pd.DataFrame({"tick": T["tick"],
                           "Smart targeting": T["smart_value"].cumsum() / T["smart_users"].cumsum(),
                           "Blanket 20% off": T["blanket_value"].cumsum() / T["blanket_users"].cumsum()})
        f1 = px.line(cv, x="tick", y=["Smart targeting", "Blanket 20% off"],
                     color_discrete_map=ui.POLICY_COLOR, labels={"value": "USD per shopper", "variable": ""})
        f1.update_traces(line_width=2.5)
        a.plotly_chart(fig_style(f1, 300, "Running value per shopper (profit + CLV of won customers)"), use_container_width=True)
        roll = T[["tick"]].copy()
        for g in ("smart", "blanket"):
            roll[g] = T[f"{g}_cost"].rolling(10, min_periods=1).sum() / T[f"{g}_rev"].rolling(10, min_periods=1).sum() * 100
        roll = roll.rename(columns={"smart": "Smart targeting", "blanket": "Blanket 20% off"})
        f2 = px.line(roll, x="tick", y=["Smart targeting", "Blanket 20% off"], color_discrete_map=ui.POLICY_COLOR,
                     labels={"value": "% of sales", "variable": ""})
        f2.add_hline(y=10, line_dash="dot", line_color=GOOD, annotation_text="target 10%", annotation_font_color=GOOD)
        f2.update_traces(line_width=2.5)
        b.plotly_chart(fig_style(f2, 300, "Promo cost as % of sales (rolling 10 ticks)"), use_container_width=True)

        a, b, c = st.columns([1.1, 1.1, 1.4])
        sm = V[V["group"] == "Smart targeting"]
        mix = sm["offer"].value_counts().reindex(list(ARM_COLOR), fill_value=0).reset_index()
        mix.columns = ["offer", "shoppers"]
        f3 = px.bar(mix, x="shoppers", y="offer", orientation="h", color="offer", color_discrete_map=ARM_COLOR, text="shoppers")
        f3.update_layout(showlegend=False, yaxis_title="", xaxis_title="shoppers (last 60 ticks)")
        a.plotly_chart(fig_style(f3, 280, "What smart targeting sends"), use_container_width=True)
        sg = V["segment"].value_counts().reindex(list(SEG_COLOR), fill_value=0).reset_index()
        sg.columns = ["segment", "shoppers"]
        f4 = px.bar(sg, x="shoppers", y="segment", orientation="h", color="segment", color_discrete_map=SEG_COLOR, text="shoppers")
        f4.update_layout(showlegend=False, yaxis_title="", xaxis_title="shoppers (last 60 ticks)")
        b.plotly_chart(fig_style(f4, 280, "Predicted segment of arriving shoppers"), use_container_width=True)
        from src.live import WH
        cov = pd.DataFrame({"warehouse": WH, "cover": [T[f"cover_{w}"].iloc[-1] for w in WH]})
        cov["status"] = np.where(cov["cover"] < 1.5, "Critical", np.where(cov["cover"] < 3, "Watch", "Healthy"))
        f5 = px.bar(cov, x="cover", y="warehouse", orientation="h", color="status", text=cov["cover"].round(1),
                    color_discrete_map={"Critical": CRIT, "Watch": WARN, "Healthy": GOOD})
        f5.update_layout(yaxis_title="", xaxis_title="ticks of demand covered by stock on hand", legend_title="")
        c.plotly_chart(fig_style(f5, 280, "Warehouse stock cover (forecast + safety stock)"), use_container_width=True)

        a, b, c = st.columns([1.5, 1.25, 1])
        a.markdown("**🧠 Live decision stream** · latest shoppers on smart targeting")
        ds = sm.tail(9)[["user", "region", "p_no_offer", "uplift", "segment", "offer", "converted"]].iloc[::-1].copy()
        ds["p_no_offer"] = (ds["p_no_offer"] * 100).round(1).astype(str) + "%"
        ds["uplift"] = (ds["uplift"] * 100).round(1).map(lambda v: f"{v:+.1f} pp")
        ds["converted"] = ds["converted"].map({1: "✅", 0: "·"})
        a.dataframe(ds.rename(columns={"p_no_offer": "buy w/o offer", "uplift": "uplift (20% off)", "converted": "bought"}),
                    hide_index=True, use_container_width=True, height=360)
        b.markdown("**💬 Review stream** · noisy, multilingual, some spam")
        html = ""
        for _, r in rv.head(7).iterrows():
            col = {"Positive": GOOD, "Negative": CRIT, "Neutral": "#8a8aa3", "Spam": WARN}[r["label"]]
            txt = str(r["text"]).replace("<", "&lt;")
            html += (f'<div class="review"><span style="color:{col};font-weight:700">{r["label"]}</span> · '
                     f'{"★" * int(r["stars"])}{"☆" * (5 - int(r["stars"]))} · <span class="small">{r["region"]} · '
                     f'{r["topic_detected"]}{" · ⚠ stars≠text" if r["rating_text_mismatch"] else ""}</span><br>{txt}</div>')
        b.markdown(html or '<div class="small">waiting for reviews…</div>', unsafe_allow_html=True)
        c.markdown("**🚨 Alerts**")
        ah = "".join(f'<div class="alert {x["kind"]}"><span class="small">{x["t"]:%H:%M:%S}</span> {x["text"]}</div>'
                     for x in sim.alerts[:8])
        c.markdown(ah or '<div class="small">All quiet.</div>', unsafe_allow_html=True)

        stock = T[["tick"] + [f"stock_{w}" for w in WH]].rename(columns=lambda x: x.replace("stock_", ""))
        f6 = px.area(stock.tail(120), x="tick", y=WH, labels={"value": "units on hand", "variable": ""})
        st.plotly_chart(fig_style(f6, 260, "Stock on hand by warehouse"), use_container_width=True)

    body()


with tabs[0]:
    live_tab()

# ============================================================ FINDINGS
with tabs[1]:
    st.markdown("## 🏆 What the analysis found, and how much improvement to expect")
    st.caption("Predicted impacts come from the tuned models plus simulation on held-out data. They assume the "
               "simulated world behaves like Temu's real one, so treat them as a direction and scale, not a promise.")
    seg_sh = P1["segment_shares"]
    precision = P1["targeting_precision"]
    lift = (seg_sh.get("High value, high risk", 0) * e_high + seg_sh.get("Mid value", 0) * e_mid +
            seg_sh.get("Low value", 0) * e_low) * precision * 100
    proj = P1["baseline_repeat90"] + lift
    k = st.columns(3)
    kpi(k[0], "P1 · 90-day repeat purchase rate", f"{P1['baseline_repeat90']:.1f}% → {proj:.1f}%",
        f"+{lift:.1f} pp predicted · target 50%", "g1")
    kpi(k[1], "P2 · promo cost as % of gross sales", f"{P2['baseline_promo_pct']:.1f}% → {P2['projected_promo_pct']:.1f}%",
        f"−{P2['promo_cost_cut_pct']:.0f}% coupon spend · orders {P2['orders_change_pct']:+.1f}%", "g2")
    kpi(k[2], "P3 · local orders hit by stockout", f"{P3['stockout_old_pct']:.1f}% → {P3['stockout_new_pct']:.1f}%",
        f"days of stock {P3['days_on_hand_old']:.0f} → {P3['days_on_hand_new']:.0f} · total cost −"
        f"{(1 - P3['total_cost_new'] / P3['total_cost_old']) * 100:.0f}%", "g3")
    st.write("")
    imp = pd.DataFrame([
        {"Problem": "P1 Customers buy once and leave", "KPI": "90-day repeat rate (%)", "Today": P1["baseline_repeat90"], "Predicted": proj, "Target (MoM)": 50},
        {"Problem": "P2 Coupons wasted on everyone", "KPI": "Promo cost % of sales", "Today": P2["baseline_promo_pct"], "Predicted": P2["projected_promo_pct"], "Target (MoM)": 10},
        {"Problem": "P3 Wrong stock in local warehouses", "KPI": "Stockout / delay rate (%)", "Today": P3["stockout_old_pct"], "Predicted": P3["stockout_new_pct"], "Target (MoM)": 5},
    ])
    fig = go.Figure()
    fig.add_bar(y=imp["KPI"], x=imp["Today"], name="Today", orientation="h", marker_color="#8a8aa3", text=imp["Today"].round(1))
    fig.add_bar(y=imp["KPI"], x=imp["Predicted"], name="Predicted after the techniques", orientation="h", marker_color=AQUA, text=imp["Predicted"].round(1))
    fig.add_scatter(y=imp["KPI"], x=imp["Target (MoM)"], mode="markers", name="Target in the Minutes of Meeting",
                    marker=dict(symbol="diamond", size=14, color=YELLOW, line=dict(color="white", width=1)))
    fig.update_layout(barmode="group", xaxis_title="%")
    st.plotly_chart(fig_style(fig, 330, "Before vs predicted after (vs the targets the group set)"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔁 Problem 1")
        b4, b3 = R["blog4"]["metrics"], R["blog3"]
        ch = A["b3_channel_ltv"].sort_values("ltv_cac_ratio")
        finding(f"<b>Only one channel pays back.</b> Over 12 months, only Organic customers bring back more than they "
                f"cost to acquire (LTV:CAC {ch['ltv_cac_ratio'].max():.1f}). {ch.iloc[0]['channel']} sits at just "
                f"{ch.iloc[0]['ltv_cac_ratio']:.2f}.", "bad")
        finding(f"<b>The churn model works.</b> It flags customers who will not order in the next 90 days with AUC "
                f"{b4['roc_auc']:.2f} and precision {b4['precision']:.0%}. Delivery delays and returns are among the strongest drivers.")
        finding(f"<b>A small group carries the revenue.</b> {A['b2_segments'].set_index('segment').loc['High value', 'share_customers_pct']:.0f}% "
                f"of customers are high value and bring {A['b2_segments'].set_index('segment').loc['High value', 'share_revenue_pct']:.0f}% of revenue.")
        finding(f"<b>Prediction:</b> repeat rate rises from {P1['baseline_repeat90']:.0f}% to {proj:.0f}%. The 50% target "
                f"needs either stronger actions (sliders on the left) or cutting the delivery delays that drive churn (Problem 3).", "warn")
    with c2:
        st.markdown("#### 🎟️ Problem 2")
        mr = A["b7_market_response"].set_index("offer")
        b8 = R["blog8"]
        finding(f"<b>Two in three coupon orders would have happened anyway.</b> Only {mr.loc['20% off', 'incremental_share_of_orders']:.0%} "
                f"of 20%-off orders are truly extra, which matches the group's assumption of 1 in 3.", "bad")
        finding(f"<b>The uplift model finds the persuadables.</b> Its scores correlate {b8['corr_true_uplift']['T-learner (per-offer models)']:.2f} "
                f"with the true uplift, and it assigns the 4 segments correctly {b8['segment_accuracy']:.0%} of the time "
                f"(random guessing: 25%).")
        b9 = R["blog9_coupons"]
        finding(f"<b>The 4-week A/B test confirms it.</b> Value per shopper rises by {money(b9['value_incl_clv_per_user']['diff'], 2)} "
                f"({ui.pfmt(b9['value_incl_clv_per_user']['p_value'])}), and promo cost falls from {b9['promo_pct_sales_control']:.0f}% "
                f"to {b9['promo_pct_sales_test']:.0f}% of sales. The change in conversion is {b9['conversion']['diff'] * 100:+.1f} pp "
                f"({ui.pfmt(b9['conversion']['p_value'])}).")
        finding(f"<b>Prediction:</b> company-wide promo cost falls from {P2['baseline_promo_pct']:.0f}% to {P2['projected_promo_pct']:.1f}% "
                f"of sales, while orders change by only {P2['orders_change_pct']:+.1f}%.", "warn" if P2['projected_promo_pct'] > 10 else "")
    with c3:
        st.markdown("#### 📦 Problem 3")
        comp = A["b6_model_comparison"].set_index("model")
        best = comp["accuracy_pct"].idxmax()
        cur = "8-week moving average (current method)"
        finding(f"<b>Better forecasts.</b> {best} reaches {comp.loc[best, 'accuracy_pct']:.1f}% accuracy "
                f"(100 − WAPE), against {comp.loc[cur, 'accuracy_pct']:.1f}% for the current 8-week average.")
        finding(f"<b>Fewer stockouts.</b> Forecast plus safety stock (z = {P3['z_chosen']:.2f}, chosen by total cost) cuts "
                f"stockouts from {P3['stockout_old_pct']:.1f}% to {P3['stockout_new_pct']:.1f}%. The pilot test is significant "
                f"({ui.pfmt(P3['pilot']['p_value'])}).")
        finding(f"<b>The cash worry is solved by ABC-XYZ.</b> Stocking only AX, AY and BX items locally covers "
                f"{P3['abc']['revenue_share_local']:.0f}% of revenue and needs {P3['abc']['cash_freed_pct']:.0f}% less "
                f"inventory cash than stocking everything.")
        finding("<b>Holding more stock costs more.</b> Days of stock on hand rise because the old method under-covered "
                "the 2–4 week lead time. Once lost sales are counted, total cost still falls.", "warn")

# ============================================================ METRICS (Blog 1)
with tabs[2]:
    st.markdown("## 📏 Know your metrics")
    blogs("Blog 1 · Know your metrics")
    m = A["b1_monthly"]
    lm = m.iloc[-1]
    k = st.columns(6)
    kpi(k[0], f"Revenue · {lm['month']}", money(lm["revenue"]), f"{lm['revenue_growth_pct']:+.1f}% MoM", "g1")
    kpi(k[1], "Active customers", f"{lm['active_customers']:,.0f}", f"{lm['new_customers']:,.0f} new", "g7")
    kpi(k[2], "North Star: repeat buyers", f"{lm['repeat_buyers']:,.0f}", "customers placing a 2nd+ order", "g3")
    kpi(k[3], "Revenue per order", money(lm["revenue_per_order"], 2), f"{lm['orders_per_customer']:.2f} orders / customer", "g2")
    kpi(k[4], "Monthly retention", pct(lm["retention_rate_pct"]), "active last month & this month", "g5")
    kpi(k[5], "Promo cost", pct(lm["promo_cost_pct"]), "of gross sales", "g6")
    a, b = st.columns(2)
    f = go.Figure()
    f.add_bar(x=m["month"], y=m["revenue"], name="Net revenue", marker_color=BLUE)
    f.update_layout(yaxis_title="USD")
    a.plotly_chart(fig_style(f, 320, "Monthly net revenue"), use_container_width=True)
    f = go.Figure()
    f.add_bar(x=m["month"], y=m["new_customers"], name="New customers", marker_color=ORANGE)
    f.add_bar(x=m["month"], y=m["existing_customers"], name="Existing customers", marker_color=AQUA)
    f.update_layout(barmode="stack", yaxis_title="active customers")
    b.plotly_chart(fig_style(f, 320, "Active customers: new vs existing"), use_container_width=True)
    a, b = st.columns(2)
    f = px.line(m, x="month", y="repeat_buyers", markers=True)
    f.update_traces(line_color=GREEN, line_width=3)
    a.plotly_chart(fig_style(f, 300, "North Star metric: monthly repeat buyers"), use_container_width=True)
    f = px.line(m.dropna(subset=["retention_rate_pct"]), x="month", y="retention_rate_pct", markers=True)
    f.update_traces(line_color=VIOLET, line_width=3)
    f.update_layout(yaxis_title="%")
    b.plotly_chart(fig_style(f, 300, "Monthly retention rate"), use_container_width=True)
    a, b = st.columns([1.4, 1])
    coh = A["b1_cohort"].set_index("cohort")
    last_p = pd.Period(m["month"].iloc[-1], "M")
    elapsed = np.array([(last_p - pd.Period(c, "M")).n for c in coh.index])
    cv_ = coh.values.astype(float)
    cv_[np.arange(cv_.shape[1])[None, :] > elapsed[:, None]] = np.nan   # months that have not happened yet
    f = px.imshow(cv_, x=[f"M{c}" for c in coh.columns], y=coh.index, color_continuous_scale=ui.SEQ_BLUE,
                  aspect="auto", labels=dict(color="% active"), text_auto=".0f")
    f.update_traces(textfont_size=9)
    a.plotly_chart(fig_style(f, 520, "Cohort retention: % of each first-order cohort active k months later"), use_container_width=True)
    r9 = A["b1_repeat90_by_cohort"]
    f = px.bar(r9, x="cohort", y="repeat90", color_discrete_sequence=[MAGENTA])
    f.add_hline(y=R["blog1"]["repeat90_pct"], line_dash="dot", line_color="white",
                annotation_text=f"avg {R['blog1']['repeat90_pct']:.1f}%")
    f.update_layout(yaxis_title="% placing a 2nd order within 90 days")
    b.plotly_chart(fig_style(f, 520, "90-day repeat rate of new buyers (the Problem 1 KPI)"), use_container_width=True)

# ============================================================ PROBLEM 1
with tabs[3]:
    st.markdown("## 🔁 Problem 1: customers buy once and leave")
    blogs("Blog 1 · Metrics", "Blog 2 · Segmentation", "Blog 3 · CLV", "Blog 4 · Churn", "Blog 5 · Next purchase day", "Blog 9 · A/B test")
    st.markdown(f"Temu spends heavily to win customers, but only **{R['blog1']['repeat90_pct']:.0f}%** of new buyers order again within 90 days.")

    st.markdown("### ① Segmentation: RFM + K-means")
    a, b, c = st.columns([1, 1, 1.2])
    el = A["b2_elbow"]
    f = go.Figure()
    f.add_scatter(x=el["k"], y=el["inertia"], mode="lines+markers", name="Inertia (elbow)", line=dict(color=BLUE, width=3))
    f.update_layout(xaxis_title="k clusters", yaxis_title="inertia")
    a.plotly_chart(fig_style(f, 300, "Elbow method: choosing k"), use_container_width=True)
    sg = A["b2_segments"].set_index("segment").loc[["Low value", "Mid value", "High value"]].reset_index()
    f = go.Figure()
    f.add_bar(x=sg["segment"], y=sg["share_customers_pct"], name="% of customers", marker_color="#8a8aa3")
    f.add_bar(x=sg["segment"], y=sg["share_revenue_pct"], name="% of revenue", marker_color=ORANGE)
    f.update_layout(barmode="group", yaxis_title="%")
    b.plotly_chart(fig_style(f, 300, "Segments: share of customers vs revenue"), use_container_width=True)
    rfm = A["b2_rfm"].sample(3000, random_state=1)
    f = px.scatter_3d(rfm, x="recency", y="frequency", z="monetary", color="segment", opacity=0.7,
                      color_discrete_map={"Low value": "#8a8aa3", "Mid value": BLUE, "High value": ORANGE})
    f.update_traces(marker_size=3)
    f.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"), legend_title="")
    c.plotly_chart(fig_style(f, 300, "RFM space (3,000 customers)"), use_container_width=True)
    st.dataframe(sg.round(1), hide_index=True, use_container_width=True)

    st.markdown("### ② Customer lifetime value")
    b3 = R["blog3"]
    a, b = st.columns([1.3, 1])
    ch = A["b3_channel_ltv"].sort_values("ltv_cac_ratio")
    f = px.bar(ch, x="ltv_cac_ratio", y="channel", orientation="h", text=ch["ltv_cac_ratio"].round(2),
               color=np.where(ch["ltv_cac_ratio"] >= 1, "pays back", "loses money"),
               color_discrete_map={"pays back": GOOD, "loses money": RED})
    f.add_vline(x=1, line_dash="dot", line_color="white", annotation_text="break-even")
    f.update_layout(xaxis_title="12-month contribution ÷ acquisition cost", yaxis_title="", legend_title="")
    a.plotly_chart(fig_style(f, 330, "LTV : CAC by acquisition channel"), use_container_width=True)
    m3 = b3["metrics"]
    with b:
        k = st.columns(2)
        kpi(k[0], "CLV class model", b3["best_model"], f"6-month CLV in 3 classes · {b3['n_customers']:,} customers", "g5")
        kpi(k[1], "Macro F1", f"{m3['f1_macro']:.2f}", f"majority-class baseline {m3['majority_baseline_f1_macro']:.2f}", "g3")
        bands = A["b3_bands"]
        st.dataframe(bands[["name", "count", "min", "mean", "max"]].rename(columns={"name": "class", "count": "customers",
                     "min": "min $", "mean": "avg $", "max": "max $"}).round(1), hide_index=True, use_container_width=True)
        st.caption(f"Accuracy {m3['accuracy']:.2f} · balanced accuracy {m3['balanced_accuracy']:.2f} "
                   f"(baseline {m3['majority_baseline_balanced_accuracy']:.2f}). Features: the blog's 3-month RFM plus full purchase history.")

    st.markdown("### ③ Churn prediction")
    b4 = R["blog4"]
    m4 = b4["metrics"]
    k = st.columns(5)
    kpi(k[0], "Best model", b4["best_model"], f"{b4['n_customers']:,} recently active customers", "g1")
    kpi(k[1], "ROC-AUC", f"{m4['roc_auc']:.3f}", "0.5 = coin flip", "g3")
    kpi(k[2], "Accuracy", f"{m4['accuracy']:.1%}", f"baseline {m4['majority_baseline_accuracy']:.1%}", "g7")
    kpi(k[3], "Precision", f"{m4['precision']:.1%}", "flagged customers who really churn", "g2")
    kpi(k[4], "Recall", f"{m4['recall']:.1%}", "churners we catch", "g4")
    a, b, c = st.columns(3)
    dec = A["b4_deciles"]
    f = go.Figure()
    f.add_bar(x=dec["decile"], y=dec["actual"] * 100, name="actual churn", marker_color=RED)
    f.add_scatter(x=dec["decile"], y=dec["predicted"] * 100, name="predicted", mode="lines+markers", line=dict(color=YELLOW, width=3))
    f.update_layout(xaxis_title="risk decile (1 = safest)", yaxis_title="% churn")
    a.plotly_chart(fig_style(f, 320, "Predicted vs actual churn by risk decile"), use_container_width=True)
    im = pd.DataFrame(b4["importance"]).iloc[::-1]
    f = px.bar(im, x="importance", y="feature", orientation="h", color_discrete_sequence=[VIOLET])
    b.plotly_chart(fig_style(f, 320, "What drives churn (feature importance)"), use_container_width=True)
    cm = np.array(b4["confusion"])
    f = px.imshow(cm, x=["pred: stays", "pred: churns"], y=["stays", "churns"], text_auto=True, color_continuous_scale=ui.SEQ_BLUE)
    f.update_layout(coloraxis_showscale=False)
    c.plotly_chart(fig_style(f, 320, "Confusion matrix (test set)"), use_container_width=True)

    st.markdown("### ④ Next purchase day")
    b5 = R["blog5"]
    m5 = b5["metrics"]
    a, b = st.columns([1, 1.2])
    with a:
        k = st.columns(2)
        kpi(k[0], "Best model", b5["best_model"], "classes: 0–20 / 21–49 / 50+ days", "g7")
        kpi(k[1], "Macro F1", f"{m5['f1_macro']:.2f}", f"baseline {m5['majority_baseline_f1_macro']:.2f} · accuracy {m5['accuracy']:.0%}", "g5")
        cd = pd.Series(b5["class_dist"]).reset_index()
        cd.columns = ["class", "customers"]
        f = px.bar(cd, x="class", y="customers", color="class", color_discrete_sequence=[RED, YELLOW, GREEN])
        f.update_layout(showlegend=False)
        st.plotly_chart(fig_style(f, 250, "When do customers with 3+ orders buy next?"), use_container_width=True)
    cm = np.array(b5["confusion"])
    f = px.imshow(cm, x=[f"pred {l}" for l in b5["labels"]], y=b5["labels"], text_auto=True, color_continuous_scale=ui.SEQ_BLUE)
    f.update_layout(coloraxis_showscale=False)
    b.plotly_chart(fig_style(f, 380, "Confusion matrix · next purchase day class"), use_container_width=True)

    st.markdown("### ⑤ Retention plan and its predicted impact")
    seg_sh = P1["segment_shares"]
    lift = (seg_sh.get("High value, high risk", 0) * e_high + seg_sh.get("Mid value", 0) * e_mid + seg_sh.get("Low value", 0) * e_low) * P1["targeting_precision"] * 100
    plan = pd.DataFrame([
        {"group": "High value, high risk", "action": "Loyalty reward + faster local delivery", "share": seg_sh.get("High value, high risk", 0), "effect_pp": e_high * 100},
        {"group": "High value, low risk (no action)", "action": "Nothing: they already come back", "share": seg_sh.get("High value, low risk (no action)", 0), "effect_pp": 0},
        {"group": "Mid value", "action": "Personalised picks timed to the predicted next-purchase day", "share": seg_sh.get("Mid value", 0), "effect_pp": e_mid * 100},
        {"group": "Low value", "action": "Free app push only, no paid discount", "share": seg_sh.get("Low value", 0), "effect_pp": e_low * 100},
    ])
    plan["share"] = (plan["share"] * 100).round(1)
    a, b = st.columns([1.5, 1])
    a.dataframe(plan.rename(columns={"share": "% of customers", "effect_pp": "assumed uplift (pp)"}), hide_index=True, use_container_width=True)
    a.latex(r"\text{Repeat}_{90}^{new}= \text{Repeat}_{90}^{today} + \text{precision}\times\sum_s \text{share}_s \times \text{uplift}_s")
    a.caption(f"Targeting precision = churn model precision ({P1['targeting_precision']:.2f}). Uplift values are "
              "assumptions; change them in the sidebar.")
    f = go.Figure(go.Indicator(mode="gauge+number+delta", value=P1["baseline_repeat90"] + lift,
                               delta={"reference": P1["baseline_repeat90"], "suffix": " pp"},
                               number={"suffix": "%"},
                               gauge={"axis": {"range": [0, 70]}, "bar": {"color": AQUA},
                                      "steps": [{"range": [0, P1["baseline_repeat90"]], "color": "#2a2f52"}],
                                      "threshold": {"line": {"color": YELLOW, "width": 4}, "value": 50}}))
    b.plotly_chart(fig_style(f, 280, "Projected 90-day repeat rate (yellow = 50% target)"), use_container_width=True)
    ab = P1["ab_test"]
    finding(f"<b>A/B test design (Blog 9):</b> 10% of new buyers get the plan for 4 weeks ({ab['n_test']:,} test vs "
            f"{ab['n_control']:,} control). In simulation the repeat rate moves from {ab['p_control']:.1%} to {ab['p_test']:.1%} "
            f"(z = {ab['z']:.2f}, {ui.pfmt(ab['p_value'])}). Detecting a 2 pp lift needs at least "
            f"{ab['required_n_per_group_for_2pp']:,} users per group.")

# ============================================================ PROBLEM 2
with tabs[4]:
    st.markdown("## 🎟️ Problem 2: coupons wasted on everyone")
    blogs("Blog 7 · Market response", "Blog 8 · Uplift + cost optimisation", "Blog 9 · A/B testing")
    mr = A["b7_market_response"]
    st.markdown(f"A randomised 4-arm experiment with **{int(mr['users'].sum()):,} users**: no offer, 20% off, "
                "buy-one-get-one and free shipping, 25% of users each.")
    st.markdown("### ① Market response: does the coupon really create sales?")
    a, b = st.columns([1.2, 1])
    f = go.Figure()
    f.add_bar(x=mr["offer"], y=mr["conversion_rate"] * 100, marker_color=[ARM_COLOR[o] for o in mr["offer"]],
              text=(mr["conversion_rate"] * 100).round(1).astype(str) + "%")
    base = mr.loc[mr["offer"] == "No offer", "conversion_rate"].iloc[0] * 100
    f.add_hline(y=base, line_dash="dot", line_color="white", annotation_text="would buy anyway")
    f.update_layout(yaxis_title="conversion %")
    a.plotly_chart(fig_style(f, 330, "Conversion by offer: the grey line is the sales we'd get for free"), use_container_width=True)
    t = mr[mr["offer"] != "No offer"][["offer", "conversion_uplift_pp", "incremental_share_of_orders", "order_uplift_per_100k",
                                      "revenue_uplift_per_100k", "promo_cost_per_user", "incremental_profit_per_user", "p_value"]].copy()
    t["incremental_share_of_orders"] = (t["incremental_share_of_orders"] * 100).round(1)
    b.dataframe(t.rename(columns={"conversion_uplift_pp": "uplift (pp)", "incremental_share_of_orders": "% orders truly extra",
                                  "order_uplift_per_100k": "extra orders / 100k", "revenue_uplift_per_100k": "extra revenue / 100k",
                                  "promo_cost_per_user": "promo $/user", "incremental_profit_per_user": "Δ profit $/user"}).round(3),
                hide_index=True, use_container_width=True)
    pct20 = mr.set_index("offer").loc["20% off", "incremental_share_of_orders"]
    b.markdown(f'<div class="finding bad">Only <b>{pct20:.0%}</b> of 20%-off orders are truly extra. The rest are '
               f'customers who would have bought anyway. Every blanket offer loses short-term profit.</div>', unsafe_allow_html=True)
    by = A["b7_uplift_by_group"]
    dim = st.radio("Who reacts to which offer?", ["price_band", "recency_band", "region"], horizontal=True,
                   format_func=lambda x: {"price_band": "coupon habit", "recency_band": "recency", "region": "region"}[x])
    h = by[by["dimension"] == dim].pivot(index="group", columns="offer", values="uplift_pp")
    f = px.imshow(h.values, x=h.columns, y=h.index, text_auto=".1f", color_continuous_scale=ui.DIVERGING,
                  color_continuous_midpoint=0, aspect="auto", labels=dict(color="uplift pp"))
    st.plotly_chart(fig_style(f, 300, "Conversion uplift (pp) by group and offer"), use_container_width=True)

    st.markdown("### ② Uplift modelling: who should get a coupon?")
    b8 = R["blog8"]
    k = st.columns(4)
    kpi(k[0], "Segment accuracy", f"{b8['segment_accuracy']:.0%}", "predicted vs true segment (random = 25%)", "g3")
    kpi(k[1], "Correlation with true uplift", f"{b8['corr_true_uplift']['T-learner (per-offer models)']:.2f}",
        f"class-transformation: {b8['corr_true_uplift']['class transformation (Blog 8)']:.2f}", "g5")
    au = b8["auuc"]
    kpi(k[2], "Area under uplift curve", f"{au['T-learner (per-offer models)'] * 100:.2f}",
        f"Blog 8 model {au['class transformation (Blog 8)'] * 100:.2f} · random {au['Random targeting'] * 100:.2f}", "g1")
    kpi(k[3], "Conversion model AUC", f"{np.mean([x['auc'] for x in b8['arm_test']]):.3f}", "average over the 4 offer models", "g2")
    a, b, c = st.columns(3)
    q = A["b8_qini"]
    f = px.line(q, x="share", y=q["incremental"] * 100, color="method",
                color_discrete_map={"class transformation (Blog 8)": VIOLET, "T-learner (per-offer models)": AQUA, "Random targeting": "#8a8aa3"},
                labels={"share": "share of users targeted (ranked by score)", "y": "extra conversions per 100 users"})
    f.update_layout(legend_title="", yaxis_title="extra conversions per 100 users")
    a.plotly_chart(fig_style(f, 340, "Uplift curve: targeting the top first"), use_container_width=True)
    ts = A["b8_test_scores"].sample(6000, random_state=2)
    f = px.histogram(ts, x="uplift_t_learner", color="_segment", nbins=60, color_discrete_map=SEG_COLOR,
                     labels={"uplift_t_learner": "predicted uplift of 20% off", "_segment": "true segment"}, barmode="overlay", opacity=0.75)
    f.update_layout(legend_title="")
    b.plotly_chart(fig_style(f, 340, "Predicted uplift, coloured by true segment"), use_container_width=True)
    segs = ["Persuadable", "Sure thing", "Lost cause", "Sleeping dog"]
    f = px.imshow(np.array(b8["segment_confusion"]), x=[f"pred {s}" for s in segs], y=segs, text_auto=True, color_continuous_scale=ui.SEQ_BLUE)
    f.update_layout(coloraxis_showscale=False)
    c.plotly_chart(fig_style(f, 340, "True vs predicted segment (test users)"), use_container_width=True)
    st.markdown("""
| Segment | Behaviour | Action |
|---|---|---|
| 🟢 **Persuadable** | buys **only** with a coupon | ✅ target, with the cheapest offer that works |
| 🔵 **Sure thing** | buys anyway | ❌ no coupon (use non-cash perks) |
| ⚪ **Lost cause** | won't buy either way | ❌ skip |
| 🔴 **Sleeping dog** | buys **less** when contacted | ❌ never contact |
""")

    st.markdown("### ③ Cost optimisation: spend only where it pays")
    st.latex(r"\Delta V_a = p_a\,(m\cdot B_a + FV - c_a) - p_0\,(m\cdot AOV + FV)\qquad \text{offer } a^*=\arg\max_a\,[\Delta V_a + \lambda\,(p_a-p_0)]")
    st.caption(f"m = margin {0.25:.0%} · B = basket · c = offer cost · FV = future value of a won customer "
               f"(${P2['future_value_usd']:.2f}, from Blog 3) · λ = shadow price of an order "
               f"(${P2.get('guardrail_lambda_usd_per_order', 0):.0f}, calibrated so orders stay ≥ 97.5% of blanket)")
    pol = A["b8_policies"].copy()
    pol["highlight"] = np.where(pol["policy"] == P2["recommended_policy"], "Recommended", np.where(pol["policy"] == "Blanket 20% off", "Today", "Other"))
    a, b = st.columns([1.1, 1])
    ps_ = pol.sort_values("incremental_value_incl_clv")
    f = px.bar(ps_, x="incremental_value_incl_clv", y="policy", orientation="h", color="highlight",
               color_discrete_map={"Recommended": AQUA, "Today": ORANGE, "Other": "#8a8aa3"},
               text=[f"promo {p_:.1f}% · orders {o_:,.0f}" for p_, o_ in zip(ps_["promo_cost_pct_sales"], ps_["orders"])],
               labels={"incremental_value_incl_clv": "value incl. CLV vs no coupons, per 100k users ($)", "policy": ""})
    f.update_traces(textposition="auto", textfont_size=11)
    f.add_vline(x=0, line_color="white", line_width=1)
    f.update_layout(showlegend=False)
    a.plotly_chart(fig_style(f, 430, "Which coupon policy creates value? (teal = recommended, orange = today)"), use_container_width=True)
    show = pol[["policy", "users_contacted_pct", "orders", "promo_cost_pct_sales", "profit", "incremental_value_incl_clv",
                "incr_revenue_per_coupon_dollar"]].round(1)
    b.dataframe(show.rename(columns={"users_contacted_pct": "% contacted", "orders": "orders/100k", "promo_cost_pct_sales": "promo % sales",
                                     "profit": "profit/100k $", "incremental_value_incl_clv": "Δ value incl CLV $",
                                     "incr_revenue_per_coupon_dollar": "extra $ per coupon $"}),
                hide_index=True, use_container_width=True, height=360)
    gc = A["b8_guardrail_curve"]
    a, b = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=gc["lambda_usd_per_order"], y=gc["orders_vs_blanket"] * 100, name="orders vs blanket (%)", line=dict(color=AQUA, width=3))
    f.add_scatter(x=gc["lambda_usd_per_order"], y=gc["promo_pct_sales"], name="promo % of sales", line=dict(color=ORANGE, width=3))
    f.add_vline(x=P2.get("guardrail_lambda_usd_per_order", 0), line_dash="dot", line_color="white", annotation_text="chosen λ")
    f.update_layout(xaxis_title="λ: value given to one extra order (USD)", yaxis_title="%")
    a.plotly_chart(fig_style(f, 320, "Order guardrail: trading coupon cost for orders"), use_container_width=True)
    fr = A["b8_frontier"]
    f = px.line(fr, x="promo_cost_pct_sales", y="value_incl_clv_per_100k", markers=True,
                labels={"promo_cost_pct_sales": "promo cost % of sales", "value_incl_clv_per_100k": "value incl. CLV per 100k users"})
    f.update_traces(line_color=VIOLET, line_width=3)
    b.plotly_chart(fig_style(f, 320, "Spend frontier: more coupons ≠ more value"), use_container_width=True)

    st.markdown("### ④ A/B test: prove it before rollout")
    b9 = R["blog9_coupons"]
    k = st.columns(4)
    cv_ = b9["conversion"]
    kpi(k[0], "Users per group", f"{b9['n_control']:,} / {b9['n_test']:,}", "stratified by spend tier × region · 4 weeks", "g1")
    kpi(k[1], "Conversion", f"{cv_['p_control']:.1%} → {cv_['p_test']:.1%}",
        f"Δ {cv_['diff'] * 100:+.2f} pp · {ui.pfmt(cv_['p_value'])}", "g7")
    vv = b9["value_incl_clv_per_user"]
    kpi(k[2], "Value per user", f"{money(vv['mean_control'], 2)} → {money(vv['mean_test'], 2)}",
        f"95% CI [{money(vv['ci_low'], 2)}, {money(vv['ci_high'], 2)}] · {ui.pfmt(vv['p_value'])}", "g3")
    kpi(k[3], "Promo cost % of sales", f"{b9['promo_pct_sales_control']:.1f}% → {b9['promo_pct_sales_test']:.1f}%",
        f"${b9['rev_per_coupon_dollar_control']:.1f} → ${b9['rev_per_coupon_dollar_test']:.1f} revenue per coupon $", "g2")
    dd = A["b9_daily"]
    a, b = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=dd["day"], y=dd["p_value_conversion"], name="p-value: conversion", line=dict(color=YELLOW, width=3))
    f.add_scatter(x=dd["day"], y=dd["p_value_profit"], name="p-value: profit per user", line=dict(color=AQUA, width=3))
    f.add_hline(y=0.05, line_dash="dot", line_color=RED, annotation_text="α = 0.05")
    f.update_layout(xaxis_title="day of test", yaxis_title="p-value", yaxis_type="log")
    a.plotly_chart(fig_style(f, 320, "Why we wait 4 weeks: p-values move day to day"), use_container_width=True)
    f = go.Figure()
    f.add_scatter(x=dd["day"], y=dd["profit_control"], name="Blanket 20% off", line=dict(color=ORANGE, width=3))
    f.add_scatter(x=dd["day"], y=dd["profit_test"], name="Smart targeting", line=dict(color=AQUA, width=3))
    f.update_layout(xaxis_title="day of test", yaxis_title="cumulative profit per user ($)")
    b.plotly_chart(fig_style(f, 320, "Short-term profit per user over the test"), use_container_width=True)
    st.caption(f"Sample size to detect a 1 pp conversion change at α = 0.05 and power = 80%: "
               f"{b9['required_n_per_group_1pp']:,} per group. Smart-group offer mix: "
               + ", ".join(f"{k_} {v_:.0f}%" for k_, v_ in b9["test_offer_mix"].items()))

# ============================================================ PROBLEM 3
with tabs[5]:
    st.markdown("## 📦 Problem 3: wrong stock in local warehouses")
    blogs("Blog 6 · Predicting sales", "Blog 5 · Next purchase day", "Blog 2 · Segmentation", "ABC-XYZ (added)", "Safety stock (added)", "Blog 9 · Pilot test")
    st.markdown("### ① Demand forecasting")
    comp = A["b6_model_comparison"].sort_values("accuracy_pct")
    a, b = st.columns([1, 1.4])
    colr = [AQUA if "LSTM" in m_ or "XGBoost" in m_ else ORANGE if "current" in m_ else "#8a8aa3" for m_ in comp["model"]]
    f = go.Figure(go.Bar(x=comp["accuracy_pct"], y=comp["model"], orientation="h", marker_color=colr,
                         text=comp["accuracy_pct"].round(1).astype(str) + "%"))
    f.update_layout(xaxis_title="forecast accuracy = 100 − WAPE (%)", xaxis_range=[40, 100])
    a.plotly_chart(fig_style(f, 330, "One-week-ahead accuracy on the last 26 weeks"), use_container_width=True)
    fdf = A["b6_forecasts"]
    whs = sorted(fdf["warehouse"].unique())
    wsel = b.selectbox("Warehouse", whs, index=whs.index("EU-Germany") if "EU-Germany" in whs else 0)
    hist = A["b6_history"]
    hh = hist[hist["warehouse"] == wsel]
    fw = fdf[fdf["warehouse"] == wsel].groupby("week").sum(numeric_only=True).reset_index()
    f = go.Figure()
    f.add_scatter(x=hh["week"], y=hh["units"], name="actual demand", line=dict(color="#c9c9de", width=2))
    best_m = A["b6_model_comparison"].sort_values("WAPE_pct").iloc[0]["model"]
    f.add_scatter(x=fw["week"], y=fw[best_m], name=best_m, line=dict(color=AQUA, width=3))
    f.add_scatter(x=fw["week"], y=fw["8-week moving average (current method)"], name="current method", line=dict(color=ORANGE, width=2, dash="dot"))
    f.update_layout(yaxis_title="units / week")
    b.plotly_chart(fig_style(f, 330, f"{wsel}: weekly units, actual vs forecast"), use_container_width=True)
    lstm = R["blog6"].get("lstm")
    a, b = st.columns(2)
    if lstm:
        a.markdown("**LSTM tuning grid** (validation MAE, normalised)")
        a.dataframe(pd.DataFrame(lstm["grid"]).round(4), hide_index=True, use_container_width=True)
    hz = pd.DataFrame(R["blog6"]["horizon_accuracy"])
    hz["accuracy_pct"] = 100 - hz["WAPE_pct"]
    f = px.bar(hz, x="horizon_weeks", y="accuracy_pct", text=hz["accuracy_pct"].round(1), color_discrete_sequence=[VIOLET])
    f.update_layout(xaxis_title="weeks ahead", yaxis_title="accuracy %", yaxis_range=[50, 100])
    b.plotly_chart(fig_style(f, 260, "XGBoost multi-horizon accuracy (used for ordering)"), use_container_width=True)

    st.markdown("### ② Inventory policy: forecast + safety stock")
    ps = A["p3_policy_summary"]
    old, new = ps[ps["policy"].str.startswith("Current")].iloc[0], ps[ps["policy"].str.startswith("New")].iloc[0]
    k = st.columns(4)
    kpi(k[0], "Stockout rate", f"{old['stockout_rate_pct']:.1f}% → {new['stockout_rate_pct']:.1f}%", "share of demand that could not ship", "g6")
    kpi(k[1], "Fill rate", f"{old['fill_rate_pct']:.1f}% → {new['fill_rate_pct']:.1f}%", "", "g3")
    kpi(k[2], "Days of stock on hand", f"{old['days_on_hand']:.1f} → {new['days_on_hand']:.1f}", "more buffer, held deliberately", "g2")
    kpi(k[3], "Total cost (26 weeks)", f"{money(P3['total_cost_old'])} → {money(P3['total_cost_new'])}",
        f"holding ${P3['hold_cost_week']}/unit-week + stockout ${P3['stockout_cost']}/unit", "g5")
    a, b = st.columns(2)
    fr = A["p3_service_frontier"]
    nf = fr[fr["policy"] == "New method"]
    f = go.Figure()
    f.add_scatter(x=nf["days_on_hand"], y=nf["stockout_rate_pct"], mode="lines+markers+text", text=nf["service_level"],
                  textposition="top right", name="new method (z varies)", line=dict(color=AQUA, width=3))
    of = fr[fr["policy"] == "Current method"]
    f.add_scatter(x=of["days_on_hand"], y=of["stockout_rate_pct"], mode="markers+text", text=["current"], textposition="top right",
                  name="current method", marker=dict(size=14, color=ORANGE))
    f.update_layout(xaxis_title="days of stock on hand", yaxis_title="stockout %")
    a.plotly_chart(fig_style(f, 330, "Service-level frontier"), use_container_width=True)
    f = go.Figure()
    lab = np.where(fr["policy"] == "Current method", "current", fr["service_level"])
    f.add_bar(x=lab, y=fr["holding_cost"], name="holding cost", marker_color=BLUE)
    f.add_bar(x=lab, y=fr["stockout_cost"], name="stockout cost", marker_color=RED)
    f.update_layout(barmode="stack", xaxis_title="policy / service level", yaxis_title="USD over 26 weeks", xaxis_type="category")
    b.plotly_chart(fig_style(f, 330, f"Total cost picks the service level (z = {P3['z_chosen']:.2f})"), use_container_width=True)
    bw = A["p3_by_warehouse"].copy()
    bw["policy"] = np.where(bw["policy"].str.startswith("Current"), "Current method", "New method")
    f = px.bar(bw, x="warehouse", y="stockout_rate_pct", color="policy", barmode="group",
               color_discrete_map={"Current method": ORANGE, "New method": AQUA}, text=bw["stockout_rate_pct"].round(1))
    f.update_layout(yaxis_title="stockout %", legend_title="")
    st.plotly_chart(fig_style(f, 320, "Stockouts by warehouse"), use_container_width=True)

    st.markdown("### ③ ABC-XYZ: which products deserve local stock?")
    abc = A["p3_abc_xyz"]
    a, b = st.columns([1, 1.3])
    mx = A["p3_abc_matrix"].set_index("ABC").reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"]).fillna(0)
    f = px.imshow(mx.values, x=["X steady", "Y variable", "Z erratic"], y=["A top 80% revenue", "B next 15%", "C last 5%"],
                  text_auto=True, color_continuous_scale=ui.SEQ_BLUE, labels=dict(color="SKUs"))
    a.plotly_chart(fig_style(f, 330, "SKU count by class"), use_container_width=True)
    f = px.scatter(abc, x="cv", y="revenue_52w", color="class", log_y=True, hover_data=["sku", "category"],
                   color_discrete_sequence=C, labels={"cv": "coefficient of variation (σ / mean)", "revenue_52w": "revenue, 52 weeks"})
    f.add_vline(x=0.5, line_dash="dot", line_color="white")
    f.add_vline(x=1.0, line_dash="dot", line_color="white")
    b.plotly_chart(fig_style(f, 330, "Each dot is a SKU"), use_container_width=True)
    ab_ = P3["abc"]
    finding(f"Stock <b>AX, AY and BX</b> items locally ({ab_['sku_share_local']:.0f}% of SKUs, {ab_['revenue_share_local']:.0f}% of revenue). "
            f"Ship the rest cross-border. Inventory cash falls from {money(ab_['inventory_value_all_local'])} to "
            f"{money(ab_['inventory_value_selective'])} (−{ab_['cash_freed_pct']:.0f}%).")

    st.markdown("### ④ Safety-stock calculator")
    a, b = st.columns([1, 1])
    with a:
        dbar = st.number_input("Average weekly demand d", 10.0, 50000.0, 500.0, 10.0)
        sd_ = st.number_input("Std of weekly demand σd", 0.0, 20000.0, 120.0, 5.0)
        LT = st.slider("Mean lead time L (weeks)", 1.0, 8.0, 2.7, 0.1)
        sL = st.slider("Std of lead time σL (weeks)", 0.0, 3.0, 0.78, 0.01)
        svc = st.select_slider("Cycle service level", options=["80%", "85%", "90%", "95%", "97.5%", "99%"], value="95%")
    from scipy.stats import norm
    z = norm.ppf(float(svc.strip("%")) / 100)
    ss = z * np.sqrt(LT * sd_ ** 2 + dbar ** 2 * sL ** 2)
    rop = dbar * LT + ss
    with b:
        st.latex(r"SS = z\sqrt{L\,\sigma_d^2 + \bar d^{\,2}\,\sigma_L^2}\qquad ROP = \bar d\,L + SS")
        k = st.columns(3)
        kpi(k[0], "z", f"{z:.2f}", f"for {svc}", "g1")
        kpi(k[1], "Safety stock", f"{ss:,.0f}", "units", "g3")
        kpi(k[2], "Reorder point", f"{rop:,.0f}", "units", "g2")

    st.markdown("### ⑤ Repeat-buyer reorder signal (Blog 5) and pilot test (Blog 9)")
    a, b = st.columns(2)
    ro = A["b5_reorder_by_warehouse"].sort_values("p_0_20")
    f = go.Figure()
    f.add_bar(y=ro["warehouse"], x=ro["p_0_20"], name="expected within 0–20 days", orientation="h", marker_color=GREEN)
    f.add_bar(y=ro["warehouse"], x=ro["p_21_49"], name="within 21–49 days", orientation="h", marker_color=YELLOW)
    f.update_layout(barmode="stack", xaxis_title="expected repeat orders")
    a.plotly_chart(fig_style(f, 300, "Predicted repeat orders by warehouse"), use_container_width=True)
    pl = P3["pilot"]
    with b:
        st.markdown(f"**Design:** {pl['design']}.")
        k = st.columns(2)
        kpi(k[0], "Stockout share of orders", f"{pl['p_control']:.1%} → {pl['p_test']:.1%}", "current lines → pilot lines", "g6")
        kpi(k[1], "Two-proportion z-test", f"z = {pl['z']:.1f}", f"{ui.pfmt(pl['p_value'])} · 95% CI [{pl['ci_low'] * 100:.1f}, {pl['ci_high'] * 100:.1f}] pp", "g3")
        st.caption("Randomising lines inside each warehouse keeps shocks such as the EU duty change or the GCC "
                   "Ramadan peak in both groups, so the comparison stays fair.")

# ============================================================ REVIEWS
with tabs[6]:
    st.markdown("## 💬 Voice of the customer: noisy reviews")
    st.caption("4,000 simulated reviews with typos, ALL CAPS, emojis, Arabic/Spanish/Portuguese/German/French snippets, "
               "sarcasm, spam and stars that contradict the text, cleaned and scored with a lexicon model.")
    rv = A["reviews_history"]
    rr = R["reviews"]
    k = st.columns(4)
    kpi(k[0], "Reviews", f"{rr['n']:,}", "May–Oct 2026", "g1")
    kpi(k[1], "Negative", pct(rr["negative_pct"]), "after spam removal", "g6")
    kpi(k[2], "Spam filtered", pct(rr["spam_pct"]), "links, bots, gibberish", "g2")
    kpi(k[3], "Stars ≠ text", pct(rr["mismatch_pct"]), f"topic detection accuracy {rr['topic_accuracy_pct']:.0f}%", "g5")
    a, b = st.columns(2)
    clean = rv[~rv["spam_flag"]]
    tl = clean.groupby(["topic_detected", "label"]).size().reset_index(name="reviews")
    f = px.bar(tl, x="reviews", y="topic_detected", color="label", orientation="h",
               color_discrete_map={"Positive": GOOD, "Neutral": "#8a8aa3", "Negative": CRIT})
    f.update_layout(yaxis_title="", legend_title="")
    a.plotly_chart(fig_style(f, 340, "Sentiment by topic: what customers complain about"), use_container_width=True)
    rg = clean.groupby("region")["sentiment"].mean().sort_values().reset_index()
    f = px.bar(rg, x="sentiment", y="region", orientation="h", text=rg["sentiment"].round(2), color_discrete_sequence=[BLUE])
    f.update_layout(yaxis_title="", xaxis_title="average sentiment (−1 to +1)")
    b.plotly_chart(fig_style(f, 340, "Average sentiment by region"), use_container_width=True)
    finding("<b>How this links to the three problems:</b> delivery and stock complaints feed Problem 3, coupon "
            "and notification complaints feed Problem 2 (they are the sleeping dogs), and every negative review "
            "is a churn signal for Problem 1.")
    st.markdown("**Examples of the noise the model has to handle**")
    ex = pd.concat([rv[rv["spam_flag"]].head(2), rv[rv["rating_text_mismatch"]].head(3),
                    rv[rv["text"].str.contains("🙄")].head(2), rv[rv["text"].str.isupper()].head(2)])
    st.dataframe(ex[["stars", "text", "label", "sentiment", "topic_detected", "spam_flag", "rating_text_mismatch"]].round(2),
                 hide_index=True, use_container_width=True)

# ============================================================ MODEL LAB
with tabs[7]:
    st.markdown("## 🧪 Model lab: hyperparameter tuning")
    st.markdown("Every model family is scored twice with k-fold cross-validation: once with **default** settings and once "
                "after a **RandomizedSearchCV** over its hyperparameters. The best tuned model is then checked on a "
                "**held-out test set** it never saw.")
    tasks = {"Blog 3 · CLV class (macro-F1)": R["blog3"]["tuning"], "Blog 4 · Churn (ROC-AUC)": R["blog4"]["tuning"],
             "Blog 5 · Next purchase day (macro-F1)": R["blog5"]["tuning"],
             "Blog 8 · Uplift class-transformation (−log loss)": R["blog8"]["class_transform_tuning"]}
    for arm, rows in R["blog8"]["arm_tuning"].items():
        tasks[f"Blog 8 · Conversion model, arm '{arm}' (−log loss)"] = rows
    allrows = []
    for t_, rows in tasks.items():
        for r_ in rows:
            allrows.append({"task": t_, **{k_: v_ for k_, v_ in r_.items() if k_ != "best_params"},
                            "best_params": ", ".join(f"{k_}={v_}" for k_, v_ in r_["best_params"].items())})
    lab = pd.DataFrame(allrows)
    sel = st.selectbox("Task", list(tasks))
    sub = lab[lab["task"] == sel]
    f = go.Figure()
    f.add_bar(x=sub["model"], y=sub["cv_default"], name="default", marker_color="#8a8aa3")
    f.add_bar(x=sub["model"], y=sub["cv_tuned"], name="tuned", marker_color=AQUA)
    lo, hi = min(sub["cv_default"].min(), sub["cv_tuned"].min()), max(sub["cv_default"].max(), sub["cv_tuned"].max())
    pad = (hi - lo) * 0.6 + 1e-3
    f.update_layout(barmode="group", yaxis_range=[lo - pad, hi + pad / 3], yaxis_title="cross-validated score")
    st.plotly_chart(fig_style(f, 330, f"{sel}: default vs tuned"), use_container_width=True)
    st.dataframe(sub[["model", "cv_default", "cv_tuned", "gain", "n_iter", "folds", "seconds", "best_params"]].round(4),
                 hide_index=True, use_container_width=True)
    st.markdown("**Forecasting (Blog 6)**")
    st.dataframe(pd.DataFrame(R["blog6"]["tuning"]).assign(best_params=lambda d: d["best_params"].map(
        lambda p: ", ".join(f"{k_}={v_}" for k_, v_ in p.items()))).round(4), hide_index=True, use_container_width=True)
    st.dataframe(A["b6_model_comparison"].round(2), hide_index=True, use_container_width=True)
    st.markdown("**Held-out test results of the chosen models**")
    tr = pd.DataFrame([
        {"model": "CLV class (Blog 3)", "chosen": R["blog3"]["best_model"], **{k_: round(v_, 3) for k_, v_ in R["blog3"]["metrics"].items()}},
        {"model": "Churn (Blog 4)", "chosen": R["blog4"]["best_model"], **{k_: round(v_, 3) for k_, v_ in R["blog4"]["metrics"].items()}},
        {"model": "Next purchase day (Blog 5)", "chosen": R["blog5"]["best_model"], **{k_: round(v_, 3) for k_, v_ in R["blog5"]["metrics"].items()}},
    ])
    st.dataframe(tr, hide_index=True, use_container_width=True)
    st.caption("Accuracy is limited by randomness built into the simulation (customers behave probabilistically), "
               "which is also true of real customers. Always compare a model with its baseline, not with 100%.")

# ============================================================ FORMULAS
with tabs[8]:
    st.markdown("## 📐 Formula sheet")
    F = [
        ("Blog 1 · Metrics", [
            (r"\text{Revenue growth}_t=\frac{Rev_t-Rev_{t-1}}{Rev_{t-1}}", "month-on-month growth"),
            (r"\text{Revenue per order}=\frac{\text{Revenue}}{\text{Orders}}", "average order value after discounts"),
            (r"\text{Retention}_t=\frac{|A_t\cap A_{t-1}|}{|A_{t-1}|}", "A = active customers in a month"),
            (r"\text{Repeat}_{90}=\frac{\#\text{ new buyers with a 2nd order}\le 90\text{ days}}{\#\text{ new buyers}}", "Problem 1 KPI"),
            (r"\text{Promo cost \%}=\frac{\sum \text{discounts}}{\sum \text{gross sales}}", "Problem 2 KPI")]),
        ("Blog 2 · Segmentation", [
            (r"R=\text{days since last order},\; F=\#\text{orders},\; M=\sum\text{revenue}", "RFM"),
            (r"\min_{C}\sum_{k}\sum_{x\in C_k}\lVert x-\mu_k\rVert^2", "K-means objective (inertia, elbow method)"),
            (r"\text{Score}=c_R+c_F+c_M\;\Rightarrow\;\text{Low}(0\!-\!2),\;\text{Mid}(3\!-\!4),\;\text{High}(5+)", "overall segment")]),
        ("Blog 3 · CLV", [
            (r"CLV_{6m}=\sum_{t\in\text{next 6 months}}\text{revenue}_t", "target, clustered into 3 classes"),
            (r"CLV_{12m}=AOV\times\text{frequency}\times m-\text{discounts}", "contribution form used for LTV:CAC"),
            (r"\text{LTV:CAC}=\frac{CLV_{12m}}{CAC}", "below 1 means the channel loses money")]),
        ("Blog 4 · Churn", [
            (r"\text{Precision}=\frac{TP}{TP+FP},\;\text{Recall}=\frac{TP}{TP+FN},\;F_1=\frac{2PR}{P+R}", "classification metrics"),
            (r"\text{AUC}=P(\hat p_{churner}>\hat p_{stayer})", "ranking quality")]),
        ("Blog 5 · Next purchase day", [
            (r"NPD=\text{first order after cutoff}-\text{last order before}", "classes 0–20, 21–49, 50+ days"),
            (r"\text{DayDiff}_k=\text{gap between the }k\text{-th last orders},\;\overline{DD},\;\sigma_{DD}", "features")]),
        ("Blog 6 · Forecasting", [
            (r"\Delta y_t=y_t-y_{t-1}", "differencing for stationarity (LSTM inputs)"),
            (r"\text{WAPE}=\frac{\sum|y-\hat y|}{\sum y},\;\text{Accuracy}=1-\text{WAPE}", "error metric"),
            (r"\hat y_{t+h}=\hat r_{t+h}\cdot\overline{y}_{t-3:t}", "XGBoost predicts the ratio to recent level (scale-free)")]),
        ("Blog 7 · Market response", [
            (r"\text{Conversion uplift}=conv_{T}-conv_{C}", "per offer"),
            (r"\text{Order uplift}=\text{Conversion uplift}\times N_T", ""),
            (r"\text{Revenue uplift}=\text{Order uplift}\times AOV", ""),
            (r"\text{Truly extra share}=\frac{conv_T-conv_C}{conv_T}", "≈ 1 in 3 for 20% off")]),
        ("Blog 8 · Uplift + cost", [
            (r"\text{Uplift}=P(TR)+P(CN)-P(TN)-P(CR)", "class transformation (equal-sized arms)"),
            (r"\hat\tau_a(x)=\hat p_a(x)-\hat p_0(x)", "T-learner, one model per offer"),
            (r"\Delta V_a=p_a(mB_a+FV-c_a)-p_0(m\,AOV+FV)", "value of sending offer a"),
            (r"a^*=\arg\max_a[\Delta V_a+\lambda(p_a-p_0)],\;\text{send if}>0", "cost optimisation with order guardrail")]),
        ("Blog 9 · A/B testing", [
            (r"z=\frac{\hat p_T-\hat p_C}{\sqrt{\hat p(1-\hat p)(1/n_T+1/n_C)}}", "two-proportion z-test"),
            (r"t=\frac{\bar x_T-\bar x_C}{\sqrt{s_T^2/n_T+s_C^2/n_C}}", "Welch t-test"),
            (r"n=\frac{(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)})^2}{\delta^2}", "sample size per group")]),
        ("Added supply-chain tools", [
            (r"\text{ABC: cumulative revenue}\le80\%\to A,\;\le95\%\to B,\;\text{else }C", ""),
            (r"\text{XYZ: }CV=\sigma/\mu\le0.5\to X,\;\le1\to Y,\;\text{else }Z", ""),
            (r"SS=z\sqrt{L\sigma_d^2+\bar d^2\sigma_L^2},\;ROP=\bar dL+SS", "safety stock and reorder point"),
            (r"\text{Total cost}=h\cdot\overline{\text{inventory}}+s\cdot\text{units short}", "chooses the service level")]),
    ]
    cols = st.columns(2)
    for i, (title, items) in enumerate(F):
        with cols[i % 2]:
            st.markdown(f'<div class="card"><b>{title}</b></div>', unsafe_allow_html=True)
            for fx, note in items:
                st.latex(fx)
                if note:
                    st.caption(note)

st.markdown('<div class="small" style="text-align:center;margin-top:24px">Temu Growth Lab · SBR group project · '
            'all data simulated · built with Python, scikit-learn, XGBoost, TensorFlow and Streamlit</div>', unsafe_allow_html=True)
