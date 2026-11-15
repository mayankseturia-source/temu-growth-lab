"""Exports everything the in-browser (claude.ai) version of the dashboard needs into one compact JSON file,
including a pool of shoppers pre-scored by the tuned uplift models for the live simulator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import data_gen as dg
from src import p2_coupons as p2

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "web_data.json"
A = {p.stem: pd.read_parquet(p) for p in ART.glob("*.parquet")}
R = json.loads((ART / "results.json").read_text())


def rnd(df, d=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(d)
    return df.replace({np.nan: None}).to_dict("list")


D = {"R": R}
m = A["b1_monthly"]
D["monthly"] = rnd(m)
coh = A["b1_cohort"].set_index("cohort")
last_p = pd.Period(m["month"].iloc[-1], "M")
elapsed = np.array([(last_p - pd.Period(c, "M")).n for c in coh.index])
cv = coh.values.astype(float)
cv[np.arange(cv.shape[1])[None, :] > elapsed[:, None]] = np.nan
D["cohort"] = {"rows": list(coh.index), "cols": [f"M{c}" for c in coh.columns],
               "z": [[None if np.isnan(v) else round(float(v), 1) for v in r] for r in cv]}
D["repeat90"] = rnd(A["b1_repeat90_by_cohort"], 2)
D["elbow"] = rnd(A["b2_elbow"], 1)
D["segments"] = rnd(A["b2_segments"], 2)
D["rfm"] = rnd(A["b2_rfm"].sample(2500, random_state=1)[["recency", "frequency", "monetary", "segment"]], 1)
D["channel_ltv"] = rnd(A["b3_channel_ltv"], 3)
D["clv_bands"] = rnd(A["b3_bands"], 1)
D["deciles"] = rnd(A["b4_deciles"], 4)
D["market"] = rnd(A["b7_market_response"], 5)
D["uplift_by"] = rnd(A["b7_uplift_by_group"], 3)
D["qini"] = rnd(A["b8_qini"], 5)
D["scores"] = rnd(A["b8_test_scores"].sample(5000, random_state=2)[["uplift_t_learner", "_segment"]], 4)
D["policies"] = rnd(A["b8_policies"], 2)
D["frontier"] = rnd(A["b8_frontier"], 3)
D["guard"] = rnd(A["b8_guardrail_curve"], 4)
D["b9daily"] = rnd(A["b9_daily"], 6)
D["fc_compare"] = rnd(A["b6_model_comparison"], 3)
f = A["b6_forecasts"]
fw = f.groupby(["warehouse", "week"])[["actual", "Ensemble: XGBoost + LSTM", "8-week moving average (current method)"]].sum().reset_index()
D["fc"] = rnd(fw, 1)
D["hist"] = rnd(A["b6_history"], 0)
D["p3sum"] = rnd(A["p3_policy_summary"], 3)
D["p3wh"] = rnd(A["p3_by_warehouse"], 3)
D["svc"] = rnd(A["p3_service_frontier"], 3)
D["abc"] = rnd(A["p3_abc_xyz"][["sku", "category", "cv", "revenue_52w", "class"]], 3)
D["abc_matrix"] = rnd(A["p3_abc_matrix"], 0)
D["reorder"] = rnd(A["b5_reorder_by_warehouse"], 1)
rv = A["reviews_history"]
clean = rv[~rv["spam_flag"]]
D["rev_topic"] = rnd(clean.groupby(["topic_detected", "label"]).size().reset_index(name="n"), 0)
D["rev_region"] = rnd(clean.groupby("region")["sentiment"].mean().sort_values().reset_index(), 3)
ex = pd.concat([rv[rv["spam_flag"]].head(2), rv[rv["rating_text_mismatch"]].head(3),
                rv[rv["text"].str.contains("🙄")].head(2), rv[rv["text"].str.isupper()].head(2)])
D["rev_examples"] = rnd(ex[["stars", "text", "label", "sentiment", "topic_detected", "spam_flag", "rating_text_mismatch"]], 2)

# ---- shopper pool for the live simulator (scored once by the tuned XGBoost/HGB uplift models)
M = joblib.load(ROOT / "models" / "uplift_models.joblib")
pop = dg.make_campaign(6000, seed=2026, assign=False)
X = p2.features(pop, M["columns"])
pool = pd.DataFrame({
    "region": pop["region"], "aov": pop["avg_order_value"].round(2), "notif": pop["notifications_per_day"],
    "sess": pop["sessions_30d"], "lo": pop["_loyalty"].round(4), "ps": pop["_price_sens"].round(4),
    "an": pop["_annoy"].round(4), "bk": pop["_basket"].round(4),
})
for i, a in enumerate(dg.ARMS):
    pool[f"q{i}"] = M["models"][a].predict_proba(X)[:, 1].round(5)   # model prediction
    pool[f"t{i}"] = pop[f"_p_{a}"].round(5)                           # true probability (simulated world)
D["pool"] = pool.to_dict("list")
D["regions"] = {r: v[0] for r, v in dg.REGIONS.items()}
D["warehouses"] = dg.WAREHOUSES
OUT.write_text(json.dumps(D, separators=(",", ":"), ensure_ascii=False, default=float))
print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
