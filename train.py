"""Builds the synthetic data, runs every analysis from the 9 blogs, tunes the models and writes the
artifacts the dashboard reads.   Run:  python train.py        (about 15-25 minutes on 2 CPUs)"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import data_gen as dg
from src import p1_retention as p1
from src import p2_coupons as p2
from src import p3_inventory as p3
from src.reviews import analyse

ROOT = Path(__file__).parent
DATA, ART, MOD = ROOT / "data", ROOT / "artifacts", ROOT / "models"
for d in (DATA, ART, MOD):
    d.mkdir(exist_ok=True)
R: dict = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "seed": dg.SEED}


def save(df: pd.DataFrame, name: str):
    df.to_parquet(ART / f"{name}.parquet", index=False)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict("records"))
    return o


T0 = time.time()
def step(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------ data
step("generating synthetic data")
cust, orders = dg.make_customers_orders()
camp = dg.make_campaign()
dem, ev = dg.make_weekly_demand()
sku, sd = dg.make_skus()
rng = np.random.default_rng(dg.SEED)
rev_hist = dg.make_reviews(4000, rng, pd.Timestamp("2026-05-01"), dg.END + pd.Timedelta(hours=23))
cust.drop(columns=[c for c in cust if c.startswith("_")]).to_parquet(DATA / "customers.parquet", index=False)
orders.to_parquet(DATA / "orders.parquet", index=False)
camp.to_parquet(DATA / "coupon_experiment.parquet", index=False)
dem.to_parquet(DATA / "warehouse_weekly_demand.parquet", index=False)
sku.to_parquet(DATA / "skus.parquet", index=False)
sd.to_parquet(DATA / "sku_weekly_demand.parquet", index=False)
R["data"] = {"customers": len(cust), "orders": len(orders), "coupon_experiment_users": len(camp),
             "warehouse_series": int(dem.groupby(["warehouse", "category"]).ngroups), "weeks": int(dem["week"].nunique()),
             "skus": len(sku), "reviews": len(rev_hist),
             "order_period": [str(orders["order_date"].min().date()), str(orders["order_date"].max().date())]}

# ------------------------------------------------------------------ Blog 1
step("Blog 1 - metrics")
monthly, cohort, r90m, repeat90 = p1.blog1_metrics(orders)
save(monthly, "b1_monthly")
save(cohort.reset_index().rename(columns=str), "b1_cohort")
save(r90m, "b1_repeat90_by_cohort")
last = monthly.iloc[-1]
R["blog1"] = {"repeat90_pct": repeat90, "promo_cost_pct": float(orders["discount"].sum() / orders["gross_value"].sum() * 100),
              "coupon_order_share_pct": float(orders["coupon_used"].mean() * 100),
              "delay_rate_pct": float(orders["delayed"].mean() * 100),
              "aov": float(orders["gross_value"].mean()), "last_month": jsonable(last.to_dict()),
              "annual_gross_sales_run_rate": float(monthly.tail(3)["gross_sales"].mean() * 12)}

# ------------------------------------------------------------------ Blog 2
step("Blog 2 - RFM segmentation")
rfm, elbow, segs, score = p1.blog2_segmentation(orders, cust)
save(rfm, "b2_rfm")
save(elbow, "b2_elbow")
save(segs, "b2_segments")
save(score, "b2_score_profile")

# ------------------------------------------------------------------ Blog 3
step("Blog 3 - CLV")
clv, clv_model, clv_cols = p1.blog3_clv(orders, cust)
ch_ltv = clv.pop("channel_ltv")
save(ch_ltv, "b3_channel_ltv")
p2.FUTURE_VALUE = float((ch_ltv["avg_contribution_12m"] * ch_ltv["customers"]).sum() / ch_ltv["customers"].sum())
save(clv.pop("bands"), "b3_bands")
R["blog3"] = jsonable(clv)

# ------------------------------------------------------------------ Blog 4
step("Blog 4 - churn")
churn, churn_model, churn_cols, risk = p1.blog4_churn(orders, cust)
save(churn.pop("risk_bands"), "b4_risk_bands")
save(churn.pop("deciles"), "b4_deciles")
R["blog4"] = jsonable(churn)

# ------------------------------------------------------------------ Blog 5
step("Blog 5 - next purchase day")
npd, npd_model, npd_cols = p1.blog5_next_purchase(orders, cust)
R["blog5"] = jsonable(npd)
# expected repeat orders per warehouse in the next 20 days (feeds Problem 3)
snap = p1._customer_features(orders[orders["order_date"] > dg.END - pd.DateOffset(months=9)], cust, dg.END,
                             start=dg.END - pd.DateOffset(months=9)).reindex(columns=npd_cols, fill_value=0)
pp = npd_model.predict_proba(snap)
wh = cust.set_index("customer_id").loc[snap.index, "warehouse"]
reorder = pd.DataFrame({"warehouse": wh.values, "p_0_20": pp[:, 2], "p_21_49": pp[:, 1]}).groupby("warehouse").sum().reset_index()
save(reorder, "b5_reorder_by_warehouse")

# ------------------------------------------------------------------ Problem 1 projection + test
step("Problem 1 - retention plan projection")
rk = risk.join(rfm.set_index("customer_id")["segment"])
grp = np.where((rk["segment"] == "High value") & (rk["p_churn"] >= 0.5), "High value, high risk",
      np.where(rk["segment"] == "High value", "High value, low risk (no action)",
      np.where(rk["segment"] == "Mid value", "Mid value", "Low value")))
shares = pd.Series(grp).value_counts(normalize=True).to_dict()
for k in p1.DEFAULT_EFFECTS:
    shares.setdefault(k, 0.0)
precision = R["blog4"]["metrics"]["precision"]
proj, lift = p1.retention_projection(repeat90, {k: shares[k] for k in p1.DEFAULT_EFFECTS}, precision)
abp1 = p1.retention_ab_test(repeat90, lift)
R["problem1"] = {"segment_shares": jsonable(shares), "effects": p1.DEFAULT_EFFECTS, "targeting_precision": precision,
                 "baseline_repeat90": repeat90, "projected_repeat90": proj, "lift_pp": lift, "ab_test": jsonable(abp1)}

# ------------------------------------------------------------------ Blog 7
step("Blog 7 - market response")
mr, mr_by = p2.blog7_market_response(camp)
save(mr, "b7_market_response")
save(mr_by, "b7_uplift_by_group")

# ------------------------------------------------------------------ Blog 8
step("Blog 8 - uplift models (tuning 5 models)")
up, arm_models, up_cols, up_te, qini = p2.blog8_uplift(camp)
R["blog8"] = jsonable(up)
save(up_te, "b8_test_scores")
save(pd.concat([pd.DataFrame({**v, "method": k}) for k, v in qini.items()]), "b8_qini")
joblib.dump({"models": arm_models, "columns": up_cols}, MOD / "uplift_models.joblib", compress=3)

step("Blog 8 - cost optimisation + policy comparison")
p2.GUARDRAIL_THR, guard_curve = p2.calibrate_guardrail(arm_models, up_cols)
save(guard_curve, "b8_guardrail_curve")
te_full = camp.set_index("user_id").loc[up_te["user_id"]].reset_index()
pred_te = {a: up_te[f"p_{a}"].values for a in dg.ARMS}
pol = p2.compare_policies(te_full, pred_te)
pol["offer_mix"] = pol["offer_mix"].map(json.dumps)
save(pol, "b8_policies")
save(p2.cost_frontier(te_full, pred_te), "b8_frontier")

# ------------------------------------------------------------------ Blog 9
step("Blog 9 - A/B test (targeted vs blanket)")
ab, ab_daily = p2.blog9_ab_test(arm_models, up_cols)
R["blog9_coupons"] = jsonable(ab)
save(ab_daily, "b9_daily")

# projection to the whole business
b = pol.set_index("policy")
blanket, rec = b.loc["Blanket 20% off"], b.loc[p2.RECOMMENDED]
cost_ratio = rec["promo_cost"] / blanket["promo_cost"]
sales_ratio = rec["gross_sales"] / blanket["gross_sales"]
base_pct = R["blog1"]["promo_cost_pct"]
R["problem2"] = {
    "baseline_promo_pct": base_pct,
    "projected_promo_pct": base_pct * cost_ratio / sales_ratio,
    "promo_cost_cut_pct": (1 - cost_ratio) * 100,
    "orders_change_pct": (rec["orders"] / blanket["orders"] - 1) * 100,
    "profit_change_per_100k": float(rec["profit"] - blanket["profit"]),
    "value_change_per_100k": float(rec["value_incl_clv"] - blanket["value_incl_clv"]),
    "future_value_usd": p2.FUTURE_VALUE,
    "annual_saving_usd_on_this_business": float(R["blog1"]["annual_gross_sales_run_rate"] * base_pct / 100 * (1 - cost_ratio)),
    "recommended_policy": p2.RECOMMENDED,
    "guardrail_lambda_usd_per_order": p2.GUARDRAIL_THR,
}

# ------------------------------------------------------------------ Blog 6 + Problem 3
step("Blog 6 - forecasting (XGBoost + LSTM tuning)")
fc, fdf, hist, mt, resid_sd, P, Pdf, scale = p3.blog6_forecast(dem, ev)
save(fc.pop("comparison"), "b6_model_comparison")
R["blog6"] = jsonable(fc)
save(fdf, "b6_forecasts")
save(hist, "b6_history")

step("Problem 3 - inventory simulation, ABC-XYZ, pilot test")
# choose safety-stock z by total cost = holding cost + stockout cost (unit value ~$10)
HOLD_COST_WEEK, STOCKOUT_COST = 0.05, 3.0  # 25%/yr of a $10 unit; lost margin + goodwill per unit short
frontier = []
for zz, svc in [(0.0, "50%"), (0.52, "70%"), (0.84, "80%"), (1.04, "85%"), (1.28, "90%"), (1.65, "95%"), (2.05, "98%")]:
    ag, _, _ = p3.simulate_inventory(P, Pdf, mt, resid_sd, scale, z=zz)
    for _, r_ in ag.iterrows():
        if r_["policy"].startswith("Current") and frontier:
            continue
        unit_weeks = r_["on_hand"] * P.shape[1] * p3.TEST_WEEKS
        frontier.append({"policy": "Current method" if r_["policy"].startswith("Current") else "New method",
                         "z": zz if r_["policy"].startswith("New") else None, "service_level": svc if r_["policy"].startswith("New") else "-",
                         "stockout_rate_pct": r_["stockout_rate_pct"], "days_on_hand": r_["days_on_hand"],
                         "holding_cost": unit_weeks * HOLD_COST_WEEK, "stockout_cost": r_["short"] * STOCKOUT_COST})
frontier = pd.DataFrame(frontier)
frontier["total_cost"] = frontier["holding_cost"] + frontier["stockout_cost"]
save(frontier, "p3_service_frontier")
nf = frontier[frontier["policy"] == "New method"]
Z_CHOSEN = float(nf.loc[nf["total_cost"].idxmin(), "z"])
agg, byw, w = p3.simulate_inventory(P, Pdf, mt, resid_sd, scale, z=Z_CHOSEN)
save(agg, "p3_policy_summary")
save(byw, "p3_by_warehouse")
save(w.groupby(["policy", "week", "warehouse"])[["demand", "filled", "short", "on_hand"]].sum().reset_index(), "p3_weekly")
pilot = p3.pilot_test(w)
abc, matrix, rev_matrix, abc_sum = p3.abc_xyz(sku, sd)
save(abc, "p3_abc_xyz")
save(matrix.reset_index(), "p3_abc_matrix")
save(rev_matrix.reset_index(), "p3_abc_rev_matrix")
a = agg.set_index("policy")
old_k, new_k = [k for k in a.index if k.startswith("Current")][0], [k for k in a.index if k.startswith("New")][0]
R["problem3"] = {"z_chosen": Z_CHOSEN, "hold_cost_week": HOLD_COST_WEEK, "stockout_cost": STOCKOUT_COST,
                 "total_cost_old": float(frontier.loc[frontier["policy"] == "Current method", "total_cost"].iloc[0]),
                 "total_cost_new": float(nf["total_cost"].min()), "stockout_old_pct": float(a.loc[old_k, "stockout_rate_pct"]), "stockout_new_pct": float(a.loc[new_k, "stockout_rate_pct"]),
                 "days_on_hand_old": float(a.loc[old_k, "days_on_hand"]), "days_on_hand_new": float(a.loc[new_k, "days_on_hand"]),
                 "pilot": jsonable(pilot), "abc": jsonable(abc_sum)}

# ------------------------------------------------------------------ reviews
step("reviews")
ra = analyse(rev_hist)
save(ra, "reviews_history")
R["reviews"] = {"n": len(ra), "spam_pct": float(ra["spam_flag"].mean() * 100),
                "negative_pct": float((ra["label"] == "Negative").mean() * 100),
                "mismatch_pct": float(ra["rating_text_mismatch"].mean() * 100),
                "topic_accuracy_pct": float((ra.loc[~ra["spam_flag"], "topic_detected"] == ra.loc[~ra["spam_flag"], "topic"]).mean() * 100)}

joblib.dump({"model": churn_model, "columns": list(churn_cols)}, MOD / "churn_model.joblib", compress=3)
(ART / "results.json").write_text(json.dumps(jsonable(R), indent=1, default=str))
step("done")
