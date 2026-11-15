"""Problem 2 - coupons wasted on everyone.  Blogs 7 (market response), 8 (uplift + cost), 9 (A/B test)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .data_gen import ARMS, GROSS_MARGIN, campaign_truth, make_campaign, true_segment
from .stats import sample_size_proportions, stratified_split, two_proportion_ztest, welch_ttest
from .tuning import tune_compare

RS = 42
FEATURES_NUM = ["recency_days", "orders_12m", "avg_order_value", "spend_12m", "coupon_use_rate",
                "cart_abandon_rate", "sessions_30d", "items_per_order", "notifications_per_day",
                "push_open_rate", "muted_notifications", "uninstalls_12m", "is_referral"]
FEATURES_CAT = ["region", "channel", "device"]
SEGMENTS = ["Persuadable", "Sure thing", "Lost cause", "Sleeping dog"]
BOGO_BASKET = 1.25
FUTURE_VALUE = 6.5  # avg 12-month contribution of a won customer (Blog 3); set from data in train.py


def features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    X = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def offer_cost(arm: str, aov):
    aov = np.asarray(aov, float)
    if arm == "20% off":
        return 0.20 * aov
    if arm == "Buy-one-get-one":
        return 0.30 * aov * BOGO_BASKET
    if arm == "Free shipping":
        return np.full_like(aov, 2.8)
    return np.zeros_like(aov)


def basket(arm, aov):
    return np.asarray(aov, float) * (BOGO_BASKET if arm == "Buy-one-get-one" else 1.0)


def incremental_profit(p: dict, aov, fv: float | None = None) -> pd.DataFrame:
    """E[value | offer a] - E[value | no offer]
       = p_a * (margin * basket_a + FV - cost_a) - p_0 * (margin * aov + FV)
       FV = future value (CLV) of a customer we win today."""
    fv = FUTURE_VALUE if fv is None else fv
    base = p["No offer"] * (GROSS_MARGIN * np.asarray(aov) + fv)
    out = {}
    for a in ARMS[1:]:
        out[a] = p[a] * (GROSS_MARGIN * basket(a, aov) + fv - offer_cost(a, aov)) - base
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Blog 7
def blog7_market_response(df: pd.DataFrame):
    rows = []
    ctrl = df[df["offer"] == "No offer"]
    for a in ARMS:
        g = df[df["offer"] == a]
        conv = g["conversion"].mean()
        aov = g.loc[g["conversion"] == 1, "revenue"].mean()
        row = {"offer": a, "users": len(g), "conversion_rate": conv, "avg_order_value": aov,
               "revenue_per_user": g["revenue"].mean(), "promo_cost_per_user": g["promo_cost"].mean()}
        if a != "No offer":
            t = two_proportion_ztest(ctrl["conversion"].sum(), len(ctrl), g["conversion"].sum(), len(g))
            # Blog 7 formulas
            row["conversion_uplift_pp"] = (conv - ctrl["conversion"].mean()) * 100        # conv_t - conv_c
            row["order_uplift_per_100k"] = (conv - ctrl["conversion"].mean()) * 100_000   # conv uplift x users
            row["revenue_uplift_per_100k"] = row["order_uplift_per_100k"] * aov           # order uplift x AOV
            row["incremental_share_of_orders"] = (conv - ctrl["conversion"].mean()) / conv
            row["p_value"] = t["p_value"]
            row["profit_per_user"] = (g["revenue"] * GROSS_MARGIN - g["promo_cost"]).mean()
        else:
            row["profit_per_user"] = (g["revenue"] * GROSS_MARGIN).mean()
        rows.append(row)
    mr = pd.DataFrame(rows)
    mr["incremental_profit_per_user"] = mr["profit_per_user"] - mr.loc[0, "profit_per_user"]
    # uplift by customer group (recency band) - who reacts?
    df = df.assign(recency_band=pd.cut(df["recency_days"], [-1, 30, 90, 180, 400],
                                       labels=["0-30 d", "31-90 d", "91-180 d", "180+ d"]).astype(str),
                   price_band=pd.qcut(df["coupon_use_rate"], 3, labels=["Low coupon users", "Mid", "Heavy coupon users"]).astype(str))
    by = []
    for col in ["recency_band", "price_band", "region"]:
        for v, g in df.groupby(col):
            c = g[g["offer"] == "No offer"]["conversion"].mean()
            for a in ARMS[1:]:
                by.append({"dimension": col, "group": v, "offer": a,
                           "uplift_pp": (g[g["offer"] == a]["conversion"].mean() - c) * 100})
    return mr, pd.DataFrame(by)


# ---------------------------------------------------------------- Blog 8
def blog8_uplift(df: pd.DataFrame, tune_iter=10):
    X = features(df)
    cols = list(X.columns)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.30, random_state=RS,
                                      stratify=df["offer"] + df["conversion"].astype(str))
    tr, te = df.iloc[idx_tr], df.iloc[idx_te]
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]

    # (a) Blog 8 class-transformation model on No offer vs 20% off (equal-sized arms)
    m2 = tr["offer"].isin(["No offer", "20% off"]).values
    t2 = tr[m2]
    is_t = (t2["offer"] == "20% off").values
    y4 = np.select([is_t & (t2["conversion"] == 1), is_t & (t2["conversion"] == 0),
                    ~is_t & (t2["conversion"] == 1)], [0, 1, 2], 3)  # TR, TN, CR, CN
    ct_rows, ct_name, ct_model = tune_compare(Xtr[m2], y4, "neg_log_loss", n_classes=4, n_iter=tune_iter, cv=3,
                                              only=["HistGradientBoosting", "XGBoost"])
    P = ct_model.predict_proba(Xte)
    uplift_ct = P[:, 0] + P[:, 3] - P[:, 1] - P[:, 2]   # P(TR) + P(CN) - P(TN) - P(CR)

    # (b) T-learner: one tuned conversion model per arm -> lets us pick the best offer per user
    arm_models, arm_rows = {}, {}
    for a in ARMS:
        ma = (tr["offer"] == a).values
        rows, name, est = tune_compare(Xtr[ma], tr.loc[ma, "conversion"].values, "neg_log_loss",
                                       n_iter=tune_iter, cv=3, only=["HistGradientBoosting", "XGBoost"],
                                       verbose=False)
        arm_models[a], arm_rows[a] = est, {"best": name, "rows": rows}
        print(f"   T-learner arm '{a}': {name}  logloss={-max(r['cv_tuned'] for r in rows):.4f}")
    pred = {a: arm_models[a].predict_proba(Xte)[:, 1] for a in ARMS}

    # held-out test metrics per arm model (on the users who actually got that arm)
    arm_test = []
    for a in ARMS:
        ma = (te["offer"] == a).values
        arm_test.append({"offer": a, "model": arm_rows[a]["best"],
                         "auc": roc_auc_score(te.loc[ma, "conversion"], pred[a][ma]),
                         "log_loss": log_loss(te.loc[ma, "conversion"], pred[a][ma]),
                         "mean_pred": float(pred[a][ma].mean()), "actual": float(te.loc[ma, "conversion"].mean())})

    uplift_t = pred["20% off"] - pred["No offer"]
    truth = te[[f"_p_{a}" for a in ARMS]].rename(columns=lambda c: c[3:])
    true_up = truth["20% off"] - truth["No offer"]

    # predicted segments vs true segments (only knowable because the data is synthetic)
    pseg = predicted_segment(pred)
    tseg = te["_segment"].values
    seg_acc = accuracy_score(tseg, pseg)
    seg_cm = confusion_matrix(tseg, pseg, labels=SEGMENTS).tolist()

    # Qini / uplift curves on the randomised test data (No offer vs 20% off users)
    qini = {"class transformation (Blog 8)": qini_curve(te, uplift_ct), "T-learner (per-offer models)": qini_curve(te, uplift_t),
            "Random targeting": qini_curve(te, np.random.default_rng(1).random(len(te)))}
    auuc = {k: float(np.trapezoid(v["incremental"], v["share"])) for k, v in qini.items()}

    out_te = te[["user_id", "offer", "conversion", "revenue", "avg_order_value", "region", "_segment"]].copy()
    out_te["uplift_class_transform"] = uplift_ct
    out_te["uplift_t_learner"] = uplift_t
    out_te["true_uplift"] = true_up.values
    out_te["predicted_segment"] = pseg
    for a in ARMS:
        out_te[f"p_{a}"] = pred[a]
    return {
        "class_transform_tuning": ct_rows, "class_transform_model": ct_name,
        "arm_tuning": {a: arm_rows[a]["rows"] for a in ARMS}, "arm_test": arm_test,
        "corr_true_uplift": {"class transformation (Blog 8)": float(np.corrcoef(uplift_ct, true_up)[0, 1]),
                             "T-learner (per-offer models)": float(np.corrcoef(uplift_t, true_up)[0, 1])},
        "segment_accuracy": float(seg_acc), "segment_confusion": seg_cm,
        "segment_f1_macro": float(f1_score(tseg, pseg, average="macro")),
        "auuc": auuc,
    }, arm_models, cols, out_te, qini


def predicted_segment(pred: dict) -> np.ndarray:
    up = np.asarray(pred["20% off"]) - np.asarray(pred["No offer"])
    pc = np.asarray(pred["No offer"])
    return np.where(up < -0.01, "Sleeping dog", np.where(up >= 0.05, "Persuadable",
                    np.where(pc >= 0.15, "Sure thing", "Lost cause")))


def qini_curve(te: pd.DataFrame, score: np.ndarray, points: int = 41):
    """Incremental conversions (per user) when targeting the top x% by score.
       inc(x) = conv_T(top x) - conv_C(top x) * n_T(top x)/n_C(top x), normalised by N."""
    m = te["offer"].isin(["No offer", "20% off"]).values
    d = pd.DataFrame({"s": score[m], "t": (te["offer"].values[m] == "20% off"), "y": te["conversion"].values[m]})
    d = d.sort_values("s", ascending=False).reset_index(drop=True)
    n = len(d)
    share, inc = [], []
    for q in np.linspace(0, 1, points):
        k = int(round(q * n))
        top = d.iloc[:k]
        nt, nc = top["t"].sum(), (~top["t"]).sum()
        yt, yc = top.loc[top["t"], "y"].sum(), top.loc[~top["t"], "y"].sum()
        val = 0.0 if nt == 0 or nc == 0 else (yt - yc * nt / nc) / n
        share.append(q)
        inc.append(val)
    return {"share": share, "incremental": inc}


# ---------------------------------------------------------------- policies + cost optimisation
def policy_actions(pred: dict, aov, policy: str, min_uplift: float = 0.02, keep_share: float = 0.8):
    n = len(aov)
    if policy == "No coupons":
        return np.array(["No offer"] * n)
    if policy.startswith("Blanket "):
        return np.array([policy.replace("Blanket ", "")] * n)
    if policy == "Random 50% get 20% off":
        return np.where(np.random.default_rng(3).random(n) < 0.5, "20% off", "No offer")
    up = pd.DataFrame({a: np.asarray(pred[a]) - np.asarray(pred["No offer"]) for a in ARMS[1:]})
    if policy == "Uplift targeting (20% off to persuadables)":
        return np.where(up["20% off"] >= min_uplift, "20% off", "No offer")
    if policy == "Uplift + cheapest offer that works":
        best = up.max(axis=1)
        cost = pd.DataFrame({a: np.asarray(pred[a]) * offer_cost(a, aov) for a in ARMS[1:]})
        ok = up.ge(keep_share * best.values[:, None]) & up.ge(min_uplift)
        cost = cost.where(ok, np.inf)
        choice = cost.idxmin(axis=1)
        return np.where(best >= min_uplift, choice, "No offer")
    if policy == "Profit-max incl. CLV (cost optimisation)":
        ip = incremental_profit(pred, aov)
        return np.where(ip.max(axis=1) > 0, ip.idxmax(axis=1), "No offer")
    if policy == RECOMMENDED:
        return guardrail_actions(pred, aov, GUARDRAIL_THR if GUARDRAIL_THR is not None else 0.0)
    raise ValueError(policy)


POLICIES = ["No coupons", "Blanket 20% off", "Blanket Buy-one-get-one", "Blanket Free shipping",
            "Random 50% get 20% off", "Uplift targeting (20% off to persuadables)",
            "Uplift + cheapest offer that works", "Profit-max incl. CLV (cost optimisation)",
            "Cost-optimised + order guardrail"]
RECOMMENDED = "Cost-optimised + order guardrail"
ORDER_GUARDRAIL = 0.975  # keep >= 97.5% of the orders blanket 20% off would bring
GUARDRAIL_THR = None    # set by calibrate_guardrail()


def guardrail_actions(pred: dict, aov, lam: float):
    """Per user choose the offer maximising  value_a + lambda * (p_a - p_0)
       (lambda = shadow price in USD of one extra order). No offer if every score <= 0."""
    ip = incremental_profit(pred, aov)
    sc = pd.DataFrame({a: ip[a].values + lam * (np.asarray(pred[a]) - np.asarray(pred["No offer"])) for a in ARMS[1:]})
    return np.where(sc.max(axis=1) > 0, sc.idxmax(axis=1), "No offer")


def calibrate_guardrail(arm_models, cols, min_ratio: float = ORDER_GUARDRAIL, n: int = 40_000, seed: int = 123):
    """Digital-twin calibration: on a FRESH simulated population (not the test set), find the smallest
       shadow price lambda whose true orders stay >= min_ratio x blanket-20%-off orders."""
    pop = make_campaign(n, seed=seed, assign=False)
    X = features(pop, cols)
    pred = {a: arm_models[a].predict_proba(X)[:, 1] for a in ARMS}
    truth = campaign_truth(pop)
    aov = pop["avg_order_value"].values
    blanket = truth["20% off"].sum()
    blanket_val = (truth["20% off"] * (GROSS_MARGIN * aov + FUTURE_VALUE - 0.2 * aov)).sum()
    curve, chosen = [], None
    for lam in np.arange(0, 60.5, 1.0):
        act = guardrail_actions(pred, aov, lam)
        v = value_policy_truth(act, truth, aov)
        ratio = v["orders"] / blanket
        curve.append({"lambda_usd_per_order": float(lam), "orders_vs_blanket": float(ratio),
                      "promo_pct_sales": float(v["promo_cost"] / v["gross_sales"] * 100),
                      "value_vs_blanket": float(v["value_incl_clv"] - blanket_val) / n * 1e5})
        if chosen is None and ratio >= min_ratio:
            chosen = float(lam)
    return (chosen if chosen is not None else 60.0), pd.DataFrame(curve)


def guardrail_threshold(pred: dict, aov, min_ratio: float = ORDER_GUARDRAIL):
    """Pick the value threshold (USD) using ONLY model predictions: the highest threshold whose
       predicted orders stay >= min_ratio x predicted orders of blanket 20% off."""
    ip = incremental_profit(pred, aov)
    best_arm, best_val = ip.idxmax(axis=1).values, ip.max(axis=1).values
    p_arm = {a: np.asarray(pred[a]) for a in ARMS}
    blanket = p_arm["20% off"].sum()
    for thr in np.linspace(1.0, -3.0, 161):
        act = np.where(best_val > thr, best_arm, "No offer")
        orders = sum(p_arm[a][act == a].sum() for a in ARMS)
        if orders >= min_ratio * blanket:
            return float(thr)
    return -3.0


def value_policy_truth(actions, truth: pd.DataFrame, aov):
    """Expected outcome per user from TRUE probabilities (available because data is simulated)."""
    aov = np.asarray(aov, float)
    p = truth.values[np.arange(len(actions)), [ARMS.index(a) for a in actions]]
    bask = np.where(actions == "Buy-one-get-one", aov * BOGO_BASKET, aov)
    cost = np.select([actions == "20% off", actions == "Buy-one-get-one", actions == "Free shipping"],
                     [0.20 * aov, 0.30 * bask, 2.8], 0.0)
    orders = p
    gross = p * bask
    promo = p * cost
    profit = gross * GROSS_MARGIN - promo
    contacted = actions != "No offer"
    return {"orders": orders.sum(), "gross_sales": gross.sum(), "promo_cost": promo.sum(), "profit": profit.sum(),
            "value_incl_clv": profit.sum() + orders.sum() * FUTURE_VALUE, "users_contacted": contacted.sum()}


def value_policy_ipw(actions, te: pd.DataFrame):
    """Off-policy estimate from the randomised experiment: average outcomes of users whose random
       arm equals the policy's choice, re-weighted by 1 / P(arm) = 4."""
    match = te["offer"].values == actions
    w = match / 0.25
    rev = te["revenue"].values
    cost = te["promo_cost"].values if "promo_cost" in te else 0
    return {"orders": (w * te["conversion"].values).sum(), "gross_sales": (w * rev).sum(),
            "promo_cost": (w * cost).sum(), "profit": (w * (rev * GROSS_MARGIN - cost)).sum()}


def compare_policies(df_test_full: pd.DataFrame, pred: dict, scale_to=100_000):
    truth = df_test_full[[f"_p_{a}" for a in ARMS]].rename(columns=lambda c: c[3:])
    aov = df_test_full["avg_order_value"].values
    n = len(df_test_full)
    base = value_policy_truth(np.array(["No offer"] * n), truth, aov)
    rows = []
    for pol in POLICIES:
        act = policy_actions(pred, aov, pol)
        v = value_policy_truth(act, truth, aov)
        ipw = value_policy_ipw(act, df_test_full)
        mix = pd.Series(act).value_counts(normalize=True).to_dict()
        f = scale_to / n
        rows.append({
            "policy": pol,
            "users_contacted_pct": v["users_contacted"] / n * 100,
            "orders": v["orders"] * f, "gross_sales": v["gross_sales"] * f, "promo_cost": v["promo_cost"] * f,
            "profit": v["profit"] * f,
            "promo_cost_pct_sales": v["promo_cost"] / v["gross_sales"] * 100,
            "incremental_orders": (v["orders"] - base["orders"]) * f,
            "incremental_profit": (v["profit"] - base["profit"]) * f,
            "value_incl_clv": v["value_incl_clv"] * f,
            "incremental_value_incl_clv": (v["value_incl_clv"] - base["value_incl_clv"]) * f,
            "incr_revenue_per_coupon_dollar": ((v["gross_sales"] - base["gross_sales"]) / v["promo_cost"]) if v["promo_cost"] > 0 else np.nan,
            "ipw_orders": ipw["orders"] * f, "ipw_profit": ipw["profit"] * f,
            "offer_mix": {k: round(float(vv) * 100, 1) for k, vv in mix.items()},
        })
    return pd.DataFrame(rows)


def cost_frontier(df_test_full, pred, n_points=25):
    """Trade-off curve: vary the minimum incremental value (incl. CLV) needed to send an offer."""
    truth = df_test_full[[f"_p_{a}" for a in ARMS]].rename(columns=lambda c: c[3:])
    aov = df_test_full["avg_order_value"].values
    n = len(df_test_full)
    out = []
    ip = incremental_profit(pred, aov)
    best_arm, best_val = ip.idxmax(axis=1).values, ip.max(axis=1).values
    for thr in np.linspace(-0.6, 0.6, n_points):
        act = np.where(best_val > thr, best_arm, "No offer")
        v = value_policy_truth(act, truth, aov)
        out.append({"min_value_usd": thr, "contacted_pct": (act != "No offer").mean() * 100,
                    "orders_per_100k": v["orders"] / n * 1e5, "promo_cost_per_100k": v["promo_cost"] / n * 1e5,
                    "profit_per_100k": v["profit"] / n * 1e5, "value_incl_clv_per_100k": v["value_incl_clv"] / n * 1e5,
                    "promo_cost_pct_sales": v["promo_cost"] / v["gross_sales"] * 100})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Blog 9: the 4-week A/B test
def blog9_ab_test(arm_models, cols, policy=None, n_users=40_000, seed=99):
    policy = policy or RECOMMENDED
    rng = np.random.default_rng(seed)
    pop = make_campaign(n_users, seed=seed, assign=False)
    X = features(pop, cols)
    pred = {a: arm_models[a].predict_proba(X)[:, 1] for a in ARMS}
    truth = campaign_truth(pop)
    # strata = value tier (spend quartile) x region group -> stratified 50/50 split
    tier = pd.qcut(pop["spend_12m"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    strata = tier + "|" + pop["region"]
    is_test = stratified_split(strata.values, rng, 0.5)
    act = np.where(is_test, policy_actions(pred, pop["avg_order_value"].values, policy), "20% off")
    p = truth.values[np.arange(n_users), [ARMS.index(a) for a in act]]
    day = rng.integers(0, 28, n_users)  # day each user entered the test
    conv = rng.random(n_users) < p
    aov = pop["avg_order_value"].values
    bask = np.where(act == "Buy-one-get-one", aov * BOGO_BASKET, aov)
    rev = np.where(conv, bask * rng.lognormal(0, 0.25, n_users), 0)
    cost = np.where(conv, np.select([act == "20% off", act == "Buy-one-get-one", act == "Free shipping"],
                                    [0.20 * rev, 0.30 * rev, 2.8], 0.0), 0.0)
    profit = rev * GROSS_MARGIN - cost
    value = profit + conv * FUTURE_VALUE
    A, B = ~is_test, is_test
    res = {
        "n_control": int(A.sum()), "n_test": int(B.sum()),
        "conversion": two_proportion_ztest(conv[A].sum(), A.sum(), conv[B].sum(), B.sum()),
        "profit_per_user": welch_ttest(profit[A], profit[B]),
        "value_incl_clv_per_user": welch_ttest(value[A], value[B]),
        "promo_cost_per_user": welch_ttest(cost[A], cost[B]),
        "revenue_per_user": welch_ttest(rev[A], rev[B]),
        "promo_pct_sales_control": cost[A].sum() / rev[A].sum() * 100,
        "promo_pct_sales_test": cost[B].sum() / rev[B].sum() * 100,
        "rev_per_coupon_dollar_control": rev[A].sum() / cost[A].sum(),
        "rev_per_coupon_dollar_test": rev[B].sum() / max(cost[B].sum(), 1e-9),
        "required_n_per_group_1pp": sample_size_proportions(conv[A].mean(), 0.01),
        "strata_balance": pd.crosstab(tier, is_test, normalize="columns").mul(100).round(1).rename(
            columns={False: "Control (blanket)", True: "Test (targeted)"}).reset_index().rename(columns={"spend_12m": "tier", "row_0": "tier"}).to_dict("records"),
        "test_offer_mix": pd.Series(act[B]).value_counts(normalize=True).mul(100).round(1).to_dict(),
    }
    # daily cumulative view (why we wait 4 weeks and don't "peek")
    daily = []
    for d in range(28):
        m = day <= d
        zt = two_proportion_ztest(conv[A & m].sum(), (A & m).sum(), conv[B & m].sum(), (B & m).sum())
        wt = welch_ttest(profit[A & m], profit[B & m])
        daily.append({"day": d + 1, "conv_control": zt["p_control"] * 100, "conv_test": zt["p_test"] * 100,
                      "profit_control": wt["mean_control"], "profit_test": wt["mean_test"],
                      "p_value_profit": wt["p_value"], "p_value_conversion": zt["p_value"]})
    return res, pd.DataFrame(daily)
