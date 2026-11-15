"""Problem 3 - wrong stock in local warehouses.
Blog 6 (sales forecasting: LSTM + tuned gradient boosting), ABC-XYZ, safety stock, inventory simulation,
Blog 9 (warehouse pilot test)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBRegressor

from .stats import two_proportion_ztest

RS = 42
TEST_WEEKS = 26
EVENTS = ["black_friday", "christmas", "back_to_school", "singles_day", "summer_sale", "ramadan"]


def _panel(dem: pd.DataFrame, ev: pd.DataFrame):
    p = dem.pivot_table(index="week", columns=["warehouse", "category"], values="units").sort_index()
    return p, ev.loc[p.index]


def _wape(y, f):
    return float(np.abs(y - f).sum() / np.abs(y).sum() * 100)


def _mape(y, f):
    return float(np.mean(np.abs(y - f) / np.maximum(y, 1)) * 100)


def _rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def build_supervised(P: np.ndarray, E: pd.DataFrame, horizons=(1, 2, 3), n_lags=12):
    """Rows = (series, origin week t, horizon h).  Target = y[t+h] / scale.  Features use info up to t."""
    T, S = P.shape
    scale = P[: T - TEST_WEEKS].mean(axis=0)
    Z = P / scale
    rows, tgt, meta = [], [], []
    ev = E[EVENTS].values
    woy = E["week_of_year"].values
    for s in range(S):
        for t in range(max(n_lags, 52), T):
            for h in horizons:
                if t + h >= T:
                    continue
                ref = Z[t - 3: t + 1, s].mean()            # recent level (4-week mean)
                lags = Z[t - n_lags + 1: t + 1, s][::-1] / ref
                ly = Z[t - 55: t - 51, s].mean()
                feats = list(lags) + [1.0, Z[t - 7: t + 1, s].mean() / ref,
                                      Z[t + h - 52, s] / ly, (Z[t - 3: t + 1, s].mean() / Z[t - 11: t - 7, s].mean()),
                                      *ev[t + h], *ev[t], woy[t + h], h, s]
                rows.append(feats)
                tgt.append(Z[t + h, s] / ref)                 # target = ratio to recent level (scale-free)
                meta.append((s, t, h))
    cols = [f"lag_{i}" for i in range(n_lags)] + ["ma4", "ma8_ratio", "last_year_seasonal_ratio", "momentum"] + \
           [f"{e}_target" for e in EVENTS] + [f"{e}_now" for e in EVENTS] + ["week_of_year", "horizon", "series"]
    meta = pd.DataFrame(meta, columns=["s", "t", "h"])
    meta["ref"] = [Z[t - 3: t + 1, s].mean() for s, t in zip(meta["s"], meta["t"])]
    return pd.DataFrame(rows, columns=cols), np.array(tgt), meta, scale


def blog6_forecast(dem: pd.DataFrame, ev: pd.DataFrame, tune_iter=14, lstm=True):
    Pdf, E = _panel(dem, ev)
    P = Pdf.values.astype(float)
    T, S = P.shape
    split = T - TEST_WEEKS
    X, y, meta, scale = build_supervised(P, E)
    origin_target = meta["t"] + meta["h"]
    tr = (origin_target < split).values
    te = (meta["t"] >= split - 1).values & (origin_target < T).values  # origins from last train week on

    # --- tuned XGBoost (time-series CV: no peeking into the future)
    t0 = time.time()
    order = np.argsort(origin_target[tr].values, kind="stable")
    Xtr, ytr = X[tr].iloc[order], y[tr][order]
    base = XGBRegressor(tree_method="hist", n_jobs=2, random_state=RS, verbosity=0)
    tss = TimeSeriesSplit(4)
    from sklearn.model_selection import cross_val_score
    default_cv = -cross_val_score(base, Xtr, ytr, cv=tss, scoring="neg_mean_absolute_error").mean()
    space = {"n_estimators": randint(200, 900), "max_depth": randint(3, 9), "learning_rate": loguniform(0.01, 0.2),
             "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
             "min_child_weight": randint(1, 20), "reg_lambda": loguniform(0.1, 30)}
    if os.environ.get("FAST"):
        tune_iter = 2
    srch = RandomizedSearchCV(base, space, n_iter=tune_iter, cv=tss, scoring="neg_mean_absolute_error",
                              random_state=RS, n_jobs=1).fit(Xtr, ytr)
    xgb = srch.best_estimator_
    tuning = [{"model": "XGBoost (direct multi-horizon)", "cv_default": float(default_cv),
               "cv_tuned": float(-srch.best_score_), "metric": "MAE (normalised units)",
               "gain": float(default_cv + srch.best_score_),
               "best_params": {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in srch.best_params_.items()},
               "n_iter": tune_iter, "folds": 4, "seconds": round(time.time() - t0, 1)}]
    print(f"   XGB forecast default MAE={default_cv:.4f} tuned={-srch.best_score_:.4f}")
    pred_xgb = xgb.predict(X[te]) * meta.loc[te, "ref"].values   # back to normalised units
    # residual std on the last train year (h=1) for safety stock
    val = tr & (origin_target >= split - 52).values & (meta["h"] == 1).values
    resid_sd = pd.Series((y[val] - xgb.predict(X[val])) * meta.loc[val, "ref"].values).groupby(meta.loc[val, "s"].values).std()

    # --- assemble one-step-ahead (h=1) test comparison incl. baselines
    m1 = te & (meta["h"] == 1).values
    mm = meta[m1].reset_index(drop=True)
    s_idx, t_idx = mm["s"].values, mm["t"].values
    actual = P[t_idx + 1, s_idx]
    fc = {
        "Naive (last week)": P[t_idx, s_idx],
        "8-week moving average (current method)": np.array([P[t - 7: t + 1, s].mean() for t, s in zip(t_idx, s_idx)]),
        "Seasonal naive (same week last year)": P[t_idx + 1 - 52, s_idx],
        "XGBoost tuned": pred_xgb[(meta[te]["h"] == 1).values] * scale[s_idx],
    }
    lstm_info = None
    if lstm:
        lstm_pred, lstm_info = _lstm(P, E, scale, split, t_idx, s_idx)
        fc["LSTM (Blog 6, tuned)"] = lstm_pred
        fc["Ensemble: XGBoost + LSTM"] = (fc["XGBoost tuned"] + lstm_pred) / 2
    comp = []
    for k, f in fc.items():
        comp.append({"model": k, "WAPE_pct": _wape(actual, f), "MAPE_pct": _mape(actual, f), "RMSE": _rmse(actual, f),
                     "bias_pct": float((f.sum() - actual.sum()) / actual.sum() * 100),
                     "accuracy_pct": 100 - _wape(actual, f)})
    comp = pd.DataFrame(comp).sort_values("WAPE_pct")
    # weekly totals per warehouse for charts
    cols = Pdf.columns
    wk = Pdf.index[t_idx + 1]
    fdf = pd.DataFrame({"week": wk, "warehouse": [cols[s][0] for s in s_idx], "category": [cols[s][1] for s in s_idx],
                        "actual": actual, **{k: v for k, v in fc.items()}})
    hist = dem.groupby(["week", "warehouse"])["units"].sum().reset_index()

    # direct multi-horizon forecasts (h=1..3) keyed by origin, used by the inventory simulation
    mt = meta[te].reset_index(drop=True)
    mt["pred"] = pred_xgb * scale[mt["s"].values]
    horizon_acc = []
    for h in (1, 2, 3):
        mh = mt[mt["h"] == h]
        a = P[mh["t"].values + h, mh["s"].values]
        horizon_acc.append({"horizon_weeks": h, "WAPE_pct": _wape(a, mh["pred"].values)})
    imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    return {
        "comparison": comp, "tuning": tuning, "lstm": lstm_info, "horizon_accuracy": horizon_acc,
        "importance": [{"feature": k, "importance": float(v)} for k, v in imp.items()],
    }, fdf, hist, mt, resid_sd, P, Pdf, scale


def _lstm(P, E, scale, split, t_idx, s_idx):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.random.set_seed(RS)
    np.random.seed(RS)
    Z = P / scale
    ev = E[EVENTS].values.astype(float)
    T, S = Z.shape

    def windows(win, t_from, t_to):
        Xs, Xe, ys, keys = [], [], [], []
        for s in range(S):
            for t in range(max(win, t_from), t_to):
                seq = Z[t - win + 1: t + 1, s] - Z[t, s]          # differenced-style input (blog: stationarity)
                Xs.append(np.c_[seq, ev[t - win + 1: t + 1]])
                Xe.append(np.r_[ev[t + 1], Z[t + 1 - 52, s] - Z[t, s] if t + 1 - 52 >= 0 else 0.0])
                ys.append(Z[t + 1, s] - Z[t, s])
                keys.append((s, t))
        return np.array(Xs), np.array(Xe), np.array(ys), keys

    results, best = [], None
    t0 = time.time()
    grid = [(12, 32, 1e-3), (12, 64, 1e-3), (8, 32, 3e-3), (16, 64, 1e-3)]
    if os.environ.get("FAST"):
        grid = grid[:1]
    for win, units, lr in grid:
        Xs, Xe, ys, _ = windows(win, 52, split - 1)
        n = len(ys)
        cut = int(n * 0.85)
        idx = np.random.default_rng(RS).permutation(n)
        trn, vl = idx[:cut], idx[cut:]
        inp_s = tf.keras.Input((win, Xs.shape[2]))
        inp_e = tf.keras.Input((Xe.shape[1],))
        h = tf.keras.layers.LSTM(units)(inp_s)
        h = tf.keras.layers.Concatenate()([h, inp_e])
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        out = tf.keras.layers.Dense(1)(h)
        model = tf.keras.Model([inp_s, inp_e], out)
        model.compile(tf.keras.optimizers.Adam(lr), loss="mae")
        cb = [tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
        hst = model.fit([Xs[trn], Xe[trn]], ys[trn], validation_data=([Xs[vl], Xe[vl]], ys[vl]),
                        epochs=60, batch_size=128, verbose=0, callbacks=cb)
        v = float(min(hst.history["val_loss"]))
        results.append({"window_weeks": win, "lstm_units": units, "learning_rate": lr, "val_MAE": v,
                        "epochs_run": len(hst.history["val_loss"])})
        print(f"   LSTM win={win} units={units} lr={lr}: val MAE={v:.4f}")
        if best is None or v < best[0]:
            best = (v, model, win)
    _, model, win = best
    Xs, Xe, _, keys = windows(win, split - 1, Z.shape[0] - 1)
    pr = model.predict([Xs, Xe], verbose=0).ravel()
    lut = {k: pr[i] for i, k in enumerate(keys)}
    pred = np.array([(lut[(s, t)] + Z[t, s]) * scale[s] for t, s in zip(t_idx, s_idx)])
    return pred, {"grid": results, "best": [r for r in results if r["val_MAE"] == best[0]][0],
                  "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- inventory simulation
LEAD_TIMES, LT_PROBS = (2, 3, 4), (0.5, 0.3, 0.2)


def simulate_inventory(P, Pdf, mt, resid_sd, scale, z=1.65, seed=5, lt_probs=LT_PROBS):
    """Weekly review, order-up-to policy, replenishment lead time 2-4 weeks (sea freight + hub).
       Old: S = MA8 x 3 weeks, no safety stock (assumes the lead time is always 2 weeks).
       New: S = forecast for review + mean lead time + SS,  SS = z * sqrt(R * sigma_e^2 + d^2 * sigma_L^2)."""
    rng = np.random.default_rng(seed)
    T, S = P.shape
    split = T - TEST_WEEKS
    fc = {(r.s, r.t, r.h): r.pred for r in mt.itertuples()}
    lt = np.array(LEAD_TIMES, float)
    mu_L = float((lt * lt_probs).sum())
    sigma_L = float(np.sqrt(((lt - mu_L) ** 2 * lt_probs).sum()))
    cover = mu_L + 1  # weeks to cover = mean lead time + 1 review week
    lts = rng.choice(list(LEAD_TIMES), size=(TEST_WEEKS + 5, S), p=list(lt_probs))
    out_rows, weekly = [], []
    for policy in ["Current method (8-wk average, no safety stock)", "New method (forecast + safety stock)"]:
        for s in range(S):
            hist = P[: split, s]
            if policy.startswith("Current"):
                S0 = hist[-8:].mean() * 3
            else:
                S0 = hist[-8:].mean() * cover + z * resid_sd.get(s, 0.1) * scale[s] * np.sqrt(cover)
            on_hand, pipeline = S0, []
            for k in range(TEST_WEEKS):
                t = split + k  # demand week
                pipeline_arr = [q for (arr, q) in pipeline if arr == k]
                on_hand += sum(pipeline_arr)
                pipeline = [(arr, q) for (arr, q) in pipeline if arr != k]
                d = P[t, s]
                filled = min(on_hand, d)
                short = d - filled
                on_hand -= filled
                # place order at end of week t (origin t) for arrival after lead time
                if policy.startswith("Current"):
                    rate = P[t - 7: t + 1, s].mean()
                    target = rate * 3
                else:
                    f = [fc.get((s, t, h), P[t, s]) for h in (1, 2, 3)]
                    dbar = np.mean(f)
                    se = resid_sd.get(s, 0.1) * scale[s]
                    ss = z * np.sqrt(cover * se ** 2 + dbar ** 2 * sigma_L ** 2)
                    target = sum(f) + dbar * (cover - 3) + ss
                inv_pos = on_hand + sum(q for _, q in pipeline)
                q = max(0.0, target - inv_pos)
                if q > 0:
                    pipeline.append((k + lts[k, s], q))
                weekly.append((policy, Pdf.columns[s][0], Pdf.columns[s][1], Pdf.index[t], d, filled, short, on_hand))
    w = pd.DataFrame(weekly, columns=["policy", "warehouse", "category", "week", "demand", "filled", "short", "on_hand"])
    agg = w.groupby("policy").agg(demand=("demand", "sum"), short=("short", "sum"), on_hand=("on_hand", "mean"),
                                  weekly_demand=("demand", "mean")).reset_index()
    agg["stockout_rate_pct"] = agg["short"] / agg["demand"] * 100
    agg["fill_rate_pct"] = 100 - agg["stockout_rate_pct"]
    agg["days_on_hand"] = agg["on_hand"] / agg["weekly_demand"] * 7
    byw = w.groupby(["policy", "warehouse"]).agg(demand=("demand", "sum"), short=("short", "sum"),
                                                 on_hand=("on_hand", "mean"), wd=("demand", "mean")).reset_index()
    byw["stockout_rate_pct"] = byw["short"] / byw["demand"] * 100
    byw["days_on_hand"] = byw["on_hand"] / byw["wd"] * 7
    return agg, byw, w


def pilot_test(w: pd.DataFrame, seed=11):
    """Blog 9 applied to warehouses: inside EVERY warehouse, half of the category lines are randomly
       switched to the new method (stratified by warehouse, so the US ramp-up or GCC Ramadan peak
       hits both groups equally). Compare share of orders hit by a stockout (two-proportion z-test)."""
    rng = np.random.default_rng(seed)
    lines = w[["warehouse", "category"]].drop_duplicates()
    lines["new"] = False
    for wh, g in lines.groupby("warehouse"):
        idx = rng.choice(g.index, size=len(g) // 2, replace=False)
        lines.loc[idx, "new"] = True
    w = w.merge(lines, on=["warehouse", "category"])
    pick = w[(w["new"] & w["policy"].str.startswith("New")) | (~w["new"] & w["policy"].str.startswith("Current"))]
    g = pick.groupby("new").agg(d=("demand", "sum"), s=("short", "sum"))
    # convert units to orders (avg 1.6 units per order) so the test counts customer orders
    n_c, x_c = g.loc[False, "d"] / 1.6, g.loc[False, "s"] / 1.6
    n_t, x_t = g.loc[True, "d"] / 1.6, g.loc[True, "s"] / 1.6
    res = two_proportion_ztest(round(x_c), round(n_c), round(x_t), round(n_t))
    res["design"] = "30 category lines on new method vs 30 on current method, 5 per warehouse each, 26 weeks"
    res["pilot_lines"] = int(lines["new"].sum())
    return res


# ---------------------------------------------------------------- ABC-XYZ + safety stock at SKU level
def abc_xyz(sku: pd.DataFrame, sd: pd.DataFrame, z=1.65, lead_weeks=2.35, sigma_L=0.48, review_weeks=1.0):
    last52 = sd[sd["week"] > sd["week"].max() - pd.Timedelta(weeks=52)]
    g = last52.groupby("sku")["units"]
    t = pd.DataFrame({"mean_weekly": g.mean(), "std_weekly": g.std(), "units_52w": g.sum()})
    t = t.join(sku.set_index("sku"))
    t["revenue_52w"] = t["units_52w"] * t["unit_price"]
    t = t.sort_values("revenue_52w", ascending=False)
    t["cum_share"] = t["revenue_52w"].cumsum() / t["revenue_52w"].sum()
    t["ABC"] = np.where(t["cum_share"] <= 0.80, "A", np.where(t["cum_share"] <= 0.95, "B", "C"))
    t["cv"] = t["std_weekly"] / t["mean_weekly"].replace(0, np.nan)        # coefficient of variation
    t["XYZ"] = np.where(t["cv"] <= 0.5, "X", np.where(t["cv"] <= 1.0, "Y", "Z"))
    t["class"] = t["ABC"] + t["XYZ"]
    # safety stock = z * sqrt(LT * sigma_d^2 + d^2 * sigma_LT^2); reorder point = d * LT + SS
    t["safety_stock"] = z * np.sqrt(lead_weeks * t["std_weekly"] ** 2 + t["mean_weekly"] ** 2 * sigma_L ** 2)
    t["reorder_point"] = t["mean_weekly"] * lead_weeks + t["safety_stock"]
    t["avg_inventory_units"] = t["mean_weekly"] * review_weeks / 2 + t["safety_stock"]
    t["inventory_value"] = t["avg_inventory_units"] * t["unit_price"]
    t["stock_locally"] = t["class"].isin(["AX", "AY", "BX"])
    matrix = t.pivot_table(index="ABC", columns="XYZ", values="unit_price", aggfunc="size", fill_value=0)
    rev_matrix = t.pivot_table(index="ABC", columns="XYZ", values="revenue_52w", aggfunc="sum", fill_value=0)
    summary = {
        "skus": int(len(t)),
        "inventory_value_all_local": float(t["inventory_value"].sum()),
        "inventory_value_selective": float(t.loc[t["stock_locally"], "inventory_value"].sum()),
        "revenue_share_local": float(t.loc[t["stock_locally"], "revenue_52w"].sum() / t["revenue_52w"].sum() * 100),
        "sku_share_local": float(t["stock_locally"].mean() * 100),
    }
    summary["cash_freed_pct"] = (1 - summary["inventory_value_selective"] / summary["inventory_value_all_local"]) * 100
    return t.reset_index(), matrix, rev_matrix, summary
