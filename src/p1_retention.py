"""Problem 1 - customers buy once and leave.  Blogs 1, 2, 3, 4, 5 (+ 9 for the retention test)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split

from .data_gen import END, GROSS_MARGIN
from .stats import sample_size_proportions, two_proportion_ztest
from .tuning import tune_compare

RS = 42


# ---------------------------------------------------------------- Blog 1
def blog1_metrics(orders: pd.DataFrame):
    o = orders.copy()
    o["month"] = o["order_date"].dt.to_period("M")
    first = o.groupby("customer_id")["order_date"].min().dt.to_period("M").rename("cohort")
    o = o.join(first, on="customer_id")
    m = o.groupby("month").agg(revenue=("net_revenue", "sum"), gross_sales=("gross_value", "sum"),
                               promo_cost=("discount", "sum"), orders=("order_id", "count"),
                               active_customers=("customer_id", "nunique"))
    m["revenue_growth_pct"] = m["revenue"].pct_change() * 100           # (Rev_t - Rev_t-1) / Rev_t-1
    m["revenue_per_order"] = m["revenue"] / m["orders"]                   # Revenue / Orders
    m["orders_per_customer"] = m["orders"] / m["active_customers"]         # Orders / Active customers
    m["promo_cost_pct"] = m["promo_cost"] / m["gross_sales"] * 100
    new = o[o["month"] == o["cohort"]].groupby("month")["customer_id"].nunique()
    m["new_customers"] = new
    m["existing_customers"] = m["active_customers"] - m["new_customers"]
    m["new_existing_ratio"] = m["new_customers"] / m["existing_customers"].replace(0, np.nan)
    m["repeat_buyers"] = o[o["order_seq"] >= 1].groupby("month")["customer_id"].nunique()  # North Star
    # Retention rate_t = |active_t ∩ active_t-1| / |active_t-1|
    act = o.groupby("month")["customer_id"].apply(set)
    ret = [np.nan] + [len(act.iloc[i] & act.iloc[i - 1]) / len(act.iloc[i - 1]) * 100 for i in range(1, len(act))]
    m["retention_rate_pct"] = ret
    m = m.reset_index()
    m["month"] = m["month"].astype(str)

    # cohort retention matrix: % of cohort active k months after first order
    o["k"] = (o["month"] - o["cohort"]).apply(lambda x: x.n)
    coh = o.groupby(["cohort", "k"])["customer_id"].nunique().unstack(fill_value=0)
    coh = coh.div(coh[0], axis=0) * 100
    coh.index = coh.index.astype(str)
    coh = coh.loc[:, coh.columns <= 12]

    # 90-day repeat rate of new buyers by cohort month
    f = o.groupby("customer_id")["order_date"].min().rename("first_date")
    o2 = o.join(f, on="customer_id")
    rep = o2[(o2["order_seq"] >= 1) & ((o2["order_date"] - o2["first_date"]).dt.days <= 90)]
    elig = f[f <= END - pd.Timedelta(days=90)]
    r90 = (elig.index.isin(rep["customer_id"].unique()))
    r90 = pd.DataFrame({"cohort": elig.dt.to_period("M").astype(str).values, "repeat90": r90})
    r90m = r90.groupby("cohort")["repeat90"].mean().mul(100).reset_index()
    return m, coh, r90m, float(r90["repeat90"].mean() * 100)


# ---------------------------------------------------------------- RFM helper
def rfm_table(orders: pd.DataFrame, asof: pd.Timestamp):
    o = orders[orders["order_date"] <= asof]
    g = o.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": (asof - g["order_date"].max()).dt.days,
        "frequency": g["order_id"].count(),
        "monetary": g["net_revenue"].sum(),
    })
    return rfm


def _ordered_kmeans(x: pd.Series, k: int, ascending: bool):
    km = KMeans(k, n_init=10, random_state=RS).fit(x.values.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers if ascending else -centers)
    rank = {c: i for i, c in enumerate(order)}
    return np.array([rank[c] for c in km.labels_])


# ---------------------------------------------------------------- Blog 2
def blog2_segmentation(orders: pd.DataFrame, customers: pd.DataFrame):
    asof = END + pd.Timedelta(days=1)
    rfm = rfm_table(orders, asof)
    # elbow + silhouette (on scaled RFM, for the explanation chart)
    z = (rfm - rfm.mean()) / rfm.std()
    sample = z.sample(min(8000, len(z)), random_state=RS)
    elbow = []
    for k in range(1, 11):
        km = KMeans(k, n_init=10, random_state=RS).fit(sample)
        sil = silhouette_score(sample, km.labels_) if k > 1 else np.nan
        elbow.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    # Blog approach: cluster each dimension (k=4), order clusters, sum into an overall score
    rfm["recency_cluster"] = _ordered_kmeans(rfm["recency"], 4, ascending=False)   # recent = high score
    rfm["frequency_cluster"] = _ordered_kmeans(rfm["frequency"], 4, ascending=True)
    rfm["revenue_cluster"] = _ordered_kmeans(rfm["monetary"], 4, ascending=True)
    rfm["overall_score"] = rfm[["recency_cluster", "frequency_cluster", "revenue_cluster"]].sum(axis=1)
    rfm["segment"] = pd.cut(rfm["overall_score"], [-1, 2, 4, 9], labels=["Low value", "Mid value", "High value"]).astype(str)
    rfm = rfm.join(customers.set_index("customer_id")[["region", "channel", "warehouse"]])
    seg = rfm.groupby("segment").agg(customers=("recency", "size"), avg_recency=("recency", "mean"),
                                     avg_frequency=("frequency", "mean"), avg_revenue=("monetary", "mean"),
                                     total_revenue=("monetary", "sum")).reset_index()
    seg["share_customers_pct"] = seg["customers"] / seg["customers"].sum() * 100
    seg["share_revenue_pct"] = seg["total_revenue"] / seg["total_revenue"].sum() * 100
    score = rfm.groupby("overall_score")[["recency", "frequency", "monetary"]].mean().reset_index()
    return rfm.reset_index(), pd.DataFrame(elbow), seg, score


def _customer_features(orders: pd.DataFrame, customers: pd.DataFrame, asof: pd.Timestamp, start=None):
    o = orders[orders["order_date"] <= asof]
    if start is not None:
        o = o[o["order_date"] >= start]
    o = o.sort_values(["customer_id", "order_date"])
    g = o.groupby("customer_id")
    f = rfm_table(o, asof)
    f["tenure_days"] = (asof - g["order_date"].min()).dt.days
    f["avg_order_value"] = g["gross_value"].mean()
    f["coupon_rate"] = g["coupon_used"].mean()
    f["avg_discount_pct"] = g["discount"].sum() / g["gross_value"].sum()
    f["delay_rate"] = g["delayed"].mean()
    f["n_delayed"] = g["delayed"].sum()
    f["return_rate"] = g["returned"].mean()
    f["avg_delivery_days"] = g["delivery_days"].mean()
    last = g.tail(1).set_index("customer_id")
    f["last_order_delayed"] = last["delayed"].astype(int)
    f["last_order_returned"] = last["returned"].astype(int)
    f["n_categories"] = g["category"].nunique()
    o = o.assign(gap=g["order_date"].diff().dt.days)
    gg = o.groupby("customer_id")["gap"]
    f["gap_mean"] = gg.mean()
    f["gap_std"] = gg.std()
    # Blog 5 "DayDiff": last three gaps
    def last_gaps(s):
        v = s.dropna().values[-3:][::-1]
        return pd.Series({"day_diff_1": v[0] if len(v) > 0 else np.nan,
                          "day_diff_2": v[1] if len(v) > 1 else np.nan,
                          "day_diff_3": v[2] if len(v) > 2 else np.nan})
    lg = o.groupby("customer_id")["gap"].apply(last_gaps).unstack()
    f = f.join(lg)
    f = f.join(customers.set_index("customer_id")[["region", "channel", "cac_usd"]])
    f = pd.get_dummies(f, columns=["region", "channel"], dtype=int)
    f["rfm_score"] = (pd.qcut(-f["recency"], 4, labels=False, duplicates="drop")
                      + pd.qcut(f["frequency"].rank(method="first"), 4, labels=False)
                      + pd.qcut(f["monetary"].rank(method="first"), 4, labels=False))
    return f.fillna(-1)


def _test_metrics(y_true, proba, labels=None):
    if proba.ndim == 1 or proba.shape[1] == 2:
        p1 = proba if proba.ndim == 1 else proba[:, 1]
        yhat = (p1 >= 0.5).astype(int)
        return {"accuracy": accuracy_score(y_true, yhat), "precision": precision_score(y_true, yhat),
                "recall": recall_score(y_true, yhat), "f1": f1_score(y_true, yhat),
                "roc_auc": roc_auc_score(y_true, p1)}, confusion_matrix(y_true, yhat).tolist()
    yhat = proba.argmax(1)
    maj = np.full_like(np.asarray(y_true), pd.Series(y_true).mode()[0])
    return {"accuracy": accuracy_score(y_true, yhat), "f1_macro": f1_score(y_true, yhat, average="macro"),
            "f1_weighted": f1_score(y_true, yhat, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(y_true, yhat),
            "majority_baseline_f1_macro": f1_score(y_true, maj, average="macro"),
            "majority_baseline_balanced_accuracy": balanced_accuracy_score(y_true, maj)}, confusion_matrix(y_true, yhat).tolist()


def _importance(est, cols, top=12):
    m = est
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    elif hasattr(m, "steps"):
        imp = np.abs(m.steps[-1][1].coef_).ravel()[: len(cols)]
    else:
        return []
    s = pd.Series(imp, index=cols).sort_values(ascending=False).head(top)
    return [{"feature": k, "importance": float(v)} for k, v in s.items()]


# ---------------------------------------------------------------- Blog 3
def blog3_clv(orders, customers, tune_iter=12):
    feat_start, feat_end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-04-30")
    tgt_start, tgt_end = pd.Timestamp("2026-05-01"), END
    window = orders[(orders["order_date"] >= feat_start) & (orders["order_date"] <= feat_end)]
    pop = window["customer_id"].unique()
    X = _customer_features(orders[orders["customer_id"].isin(pop)], customers, feat_end)
    w3 = rfm_table(window, feat_end + pd.Timedelta(days=1)).add_prefix("w3_")  # blog: 3-month RFM
    X = X.join(w3).fillna(-1)
    fut = orders[(orders["order_date"] >= tgt_start) & (orders["order_date"] <= tgt_end)]
    y6 = fut.groupby("customer_id")["net_revenue"].sum().reindex(X.index).fillna(0)
    cap = y6.quantile(0.99)
    y6c = y6.clip(upper=cap)
    km = KMeans(3, n_init=10, random_state=RS).fit(y6c.values.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel())
    rank = {c: i for i, c in enumerate(order)}
    ycls = np.array([rank[c] for c in km.labels_])
    names = ["Low CLV", "Mid CLV", "High CLV"]
    bands = pd.DataFrame({"clv_6m": y6c, "cls": ycls}).groupby("cls")["clv_6m"].agg(["min", "max", "mean", "count"])
    Xtr, Xte, ytr, yte = train_test_split(X, ycls, test_size=0.25, random_state=RS, stratify=ycls)
    rows, best_name, best = tune_compare(Xtr, ytr, "f1_macro", n_classes=3, n_iter=tune_iter)
    proba = best.predict_proba(Xte)
    met, cm = _test_metrics(yte, proba)
    met["majority_baseline_accuracy"] = float(pd.Series(yte).value_counts(normalize=True).max())

    # LTV : CAC by acquisition channel (first 12 months contribution margin)
    first = orders.groupby("customer_id")["order_date"].min()
    elig = first[first <= END - pd.Timedelta(days=365)].index
    o = orders[orders["customer_id"].isin(elig)].join(first.rename("first"), on="customer_id")
    o = o[(o["order_date"] - o["first"]).dt.days <= 365]
    o["contribution"] = o["gross_value"] * GROSS_MARGIN - o["discount"]
    ltv = o.groupby("customer_id").agg(revenue_12m=("net_revenue", "sum"), contribution_12m=("contribution", "sum"),
                                       orders_12m=("order_id", "count"))
    ltv = ltv.join(customers.set_index("customer_id")[["channel", "cac_usd"]])
    ch = ltv.groupby("channel").agg(customers=("cac_usd", "size"), avg_cac=("cac_usd", "mean"),
                                    avg_revenue_12m=("revenue_12m", "mean"),
                                    avg_contribution_12m=("contribution_12m", "mean"),
                                    avg_orders_12m=("orders_12m", "mean")).reset_index()
    # CLV formula = AOV x purchase frequency x margin (net of promo) over 12 months
    ch["ltv_cac_ratio"] = ch["avg_contribution_12m"] / ch["avg_cac"]
    return {
        "metrics": met, "confusion": cm, "labels": names, "tuning": rows, "best_model": best_name,
        "bands": bands.reset_index().assign(name=lambda d: [names[i] for i in d["cls"]]),
        "importance": _importance(best, list(X.columns)),
        "channel_ltv": ch, "n_customers": int(len(X)),
        "pred_dist": pd.Series(proba.argmax(1)).map(dict(enumerate(names))).value_counts().to_dict(),
    }, best, X.columns


# ---------------------------------------------------------------- Blog 4
def blog4_churn(orders, customers, tune_iter=15):
    cutoff = pd.Timestamp("2026-07-31")
    recent = orders[(orders["order_date"] > cutoff - pd.Timedelta(days=180)) & (orders["order_date"] <= cutoff)]
    pop = recent["customer_id"].unique()
    X = _customer_features(orders[orders["customer_id"].isin(pop)], customers, cutoff)
    fut = orders[(orders["order_date"] > cutoff) & (orders["order_date"] <= END)]
    y = (~X.index.isin(fut["customer_id"].unique())).astype(int)  # 1 = churned (no order in next 90 days)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RS, stratify=y)
    rows, best_name, best = tune_compare(Xtr, ytr, "roc_auc", n_iter=tune_iter)
    proba = best.predict_proba(Xte)
    met, cm = _test_metrics(yte, proba)
    met["churn_rate"] = float(y.mean())
    met["majority_baseline_accuracy"] = float(max(y.mean(), 1 - y.mean()))
    # risk bands on test set
    risk = pd.DataFrame({"p_churn": proba[:, 1], "churned": yte}, index=Xte.index)
    risk["risk_band"] = pd.cut(risk["p_churn"], [0, 0.4, 0.7, 1.0], labels=["Low risk", "Medium risk", "High risk"],
                               include_lowest=True).astype(str)
    band = risk.groupby("risk_band").agg(customers=("p_churn", "size"), actual_churn_rate=("churned", "mean")).reset_index()
    # calibration-style chart: deciles
    risk["decile"] = pd.qcut(risk["p_churn"].rank(method="first"), 10, labels=False) + 1
    dec = risk.groupby("decile").agg(predicted=("p_churn", "mean"), actual=("churned", "mean")).reset_index()
    return {"metrics": met, "confusion": cm, "tuning": rows, "best_model": best_name,
            "importance": _importance(best, list(X.columns)), "risk_bands": band, "deciles": dec,
            "n_customers": int(len(X))}, best, X.columns, risk


# ---------------------------------------------------------------- Blog 5
def blog5_next_purchase(orders, customers, tune_iter=12):
    cutoff = pd.Timestamp("2026-07-31")
    start9 = cutoff - pd.DateOffset(months=9)
    hist = orders[(orders["order_date"] > start9) & (orders["order_date"] <= cutoff)]
    cnt = hist.groupby("customer_id").size()
    pop = cnt[cnt >= 3].index  # blog: customers with 3+ purchases (DayDiff features)
    X = _customer_features(orders[orders["customer_id"].isin(pop)], customers, cutoff, start=start9)
    last = hist.groupby("customer_id")["order_date"].max()
    nxt = orders[orders["order_date"] > cutoff].groupby("customer_id")["order_date"].min()
    npd = (nxt.reindex(X.index) - last.reindex(X.index)).dt.days.fillna(9999)
    # classes as in the blog: 0-20 days, 21-49 days, 50+ days (incl. no purchase)
    cls = np.where(npd <= 20, 2, np.where(npd <= 49, 1, 0))
    names = ["50+ days / none", "21-49 days", "0-20 days"]
    Xtr, Xte, ytr, yte = train_test_split(X, cls, test_size=0.25, random_state=RS, stratify=cls)
    rows, best_name, best = tune_compare(Xtr, ytr, "f1_macro", n_classes=3, n_iter=tune_iter)
    proba = best.predict_proba(Xte)
    met, cm = _test_metrics(yte, proba)
    met["majority_baseline_accuracy"] = float(pd.Series(yte).value_counts(normalize=True).max())
    dist = pd.Series(cls).map(dict(enumerate(names))).value_counts().to_dict()
    return {"metrics": met, "confusion": cm, "labels": names, "tuning": rows, "best_model": best_name,
            "importance": _importance(best, list(X.columns)), "class_dist": dist,
            "n_customers": int(len(X))}, best, X.columns


# ---------------------------------------------------------------- Retention plan projection + A/B (Blog 9)
DEFAULT_EFFECTS = {  # absolute uplift in 90-day repeat probability (assumptions, editable in the app)
    "High value, high risk": 0.16,   # loyalty reward + faster local delivery
    "Mid value": 0.11,               # personalised picks timed to predicted next-purchase day
    "Low value": 0.03,               # free push message only
}


def retention_projection(baseline_repeat: float, seg_shares: dict, precision: float,
                         effects: dict = DEFAULT_EFFECTS):
    """Projected 90d repeat = baseline + sum_s share_s * effect_s * targeting precision."""
    lift = sum(seg_shares[s] * effects[s] for s in effects) * precision
    return baseline_repeat + lift * 100, lift * 100


def retention_ab_test(baseline_repeat: float, lift_pp: float, n_users: int = 40_000, seed: int = 7):
    """Simulated 4-week test: 10% of new buyers get the plan, 90% control."""
    rng = np.random.default_rng(seed)
    n_t = int(n_users * 0.10)
    n_c = n_users - n_t
    p_c = baseline_repeat / 100
    p_t = p_c + lift_pp / 100
    x_c = rng.binomial(n_c, p_c)
    x_t = rng.binomial(n_t, p_t)
    res = two_proportion_ztest(x_c, n_c, x_t, n_t)
    res.update({"n_control": n_c, "n_test": n_t,
                "required_n_per_group_for_2pp": sample_size_proportions(p_c, 0.02)})
    return res
