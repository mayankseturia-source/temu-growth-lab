"""Live data simulator: every tick new shoppers arrive, the trained uplift models decide who gets which
coupon (live A/B: half the traffic on blanket 20% off, half on smart targeting), orders hit the six local
warehouses, stock is replenished with the forecast + safety-stock rule, and noisy customer reviews stream in.
All data is SIMULATED."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import data_gen as dg
from . import p2_coupons as p2
from .reviews import analyse

WH = dg.WAREHOUSES
REGION_WH = {r: v[0] for r, v in dg.REGIONS.items()}


def build_pool(models: dict, cols: list, n: int = 30_000, seed: int = 2026) -> pd.DataFrame:
    """Score a large pool of simulated shoppers ONCE (batch prediction is fast); each tick samples from it."""
    pop = dg.make_campaign(n, seed=seed, assign=False)
    X = p2.features(pop, cols)
    for a in dg.ARMS:
        pop[f"pred_{a}"] = models[a].predict_proba(X)[:, 1]
    return pop


class LiveSim:
    def __init__(self, pool: pd.DataFrame, seed: int | None = None, guardrail: float = 25.0, fv: float = 6.4):
        self.pool = pool
        self.rng = np.random.default_rng(seed if seed is not None else int(time.time()))
        self.tick = 0
        self.start = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")
        self.guardrail, self.fv = guardrail, fv
        self.visits = []      # per-visitor decision rows (rolling)
        self.ticks = []       # per-tick aggregates
        self.reviews = pd.DataFrame()
        self.alerts = []
        share = {w: 0.0 for w in WH}
        for r, (w, sh, _) in dg.REGIONS.items():
            share[w] += sh
        units_per_tick = 70 * 0.165 * 1.65 * 6          # shoppers x conversion x units/order x network scale
        self.rate = {w: units_per_tick * share[w] for w in WH}   # EWMA demand forecast per tick
        self.err = {w: self.rate[w] * 0.35 for w in WH}
        self.stock = {w: self.rate[w] * 6 for w in WH}
        self.pipeline = []    # (arrival_tick, warehouse, qty)
        self.incident = {}    # name -> ticks remaining

    # ------------------------------------------------------------------ incidents
    def trigger(self, name: str, ticks: int = 12):
        self.incident[name] = ticks
        msg = {"spike": "🛍️ Mega-sale traffic spike started (x3 shoppers)",
               "delay": "🚚 Delivery delay storm: carrier backlog in GCC + EU",
               "blast": "📣 Marketing sent a notification blast to everyone"}[name]
        self._alert("warn", msg)

    def _alert(self, kind, text):
        self.alerts.insert(0, {"t": self.now(), "kind": kind, "text": text})
        self.alerts = self.alerts[:40]

    def now(self):
        return self.start + pd.Timedelta(seconds=int(self.tick * 3))

    # ------------------------------------------------------------------ one tick
    def step(self, traffic: float = 1.0):
        self.tick += 1
        for k in list(self.incident):
            self.incident[k] -= 1
            if self.incident[k] <= 0:
                del self.incident[k]
                self._alert("good", {"spike": "Traffic back to normal", "delay": "Carrier backlog cleared",
                                     "blast": "Notification blast ended"}[k])
        mult = traffic * (3.0 if "spike" in self.incident else 1.0) * (1 + 0.25 * np.sin(self.tick / 9))
        n = max(5, int(self.rng.poisson(70 * mult)))
        pop = self.pool.iloc[self.rng.integers(0, len(self.pool), n)].reset_index(drop=True)
        if "blast" in self.incident:   # the blast annoys people: true behaviour changes, the model doesn't know
            pop = pop.assign(notifications_per_day=pop["notifications_per_day"] + 6)
            truth = dg.campaign_truth(pop)
        else:
            truth = pop[[f"_p_{a}" for a in dg.ARMS]].rename(columns=lambda c: c[3:])
        pred = {a: pop[f"pred_{a}"].values for a in dg.ARMS}
        aov = pop["avg_order_value"].values
        smart = self.rng.random(n) < 0.5
        act_smart = p2.guardrail_actions(pred, aov, self.guardrail)
        act = np.where(smart, act_smart, "20% off")
        p = truth.values[np.arange(n), [dg.ARMS.index(a) for a in act]]
        conv = self.rng.random(n) < p
        bask = np.where(act == "Buy-one-get-one", aov * p2.BOGO_BASKET, aov)
        rev = np.where(conv, bask * self.rng.lognormal(0, 0.25, n), 0.0)
        cost = np.where(conv, np.select([act == "20% off", act == "Buy-one-get-one", act == "Free shipping"],
                                        [0.2 * rev, 0.3 * rev, 2.8], 0.0), 0.0)
        value = rev * dg.GROSS_MARGIN - cost + conv * self.fv
        seg = p2.predicted_segment(pred)
        ts = self.now()
        wh = pop["region"].map(REGION_WH).values
        df = pd.DataFrame({"tick": self.tick, "time": ts, "user": [f"LV{self.tick:04d}-{i:03d}" for i in range(n)],
                           "region": pop["region"].values, "warehouse": wh,
                           "group": np.where(smart, "Smart targeting", "Blanket 20% off"), "offer": act,
                           "p_no_offer": pred["No offer"], "p_20off": pred["20% off"],
                           "uplift": pred["20% off"] - pred["No offer"], "segment": seg,
                           "converted": conv.astype(int), "revenue": rev, "promo_cost": cost, "value": value})
        self.visits.append(df)
        self.visits = self.visits[-200:]

        # -------- warehouses: demand, stockouts, replenishment (forecast + safety stock)
        units = df[df["converted"] == 1].groupby("warehouse").size().reindex(WH, fill_value=0) * \
            self.rng.uniform(1.4, 1.9) * 6  # each order ~1.6 units; x6 = whole-network scale
        short_tot, dem_tot = 0.0, 0.0
        arrived = [x for x in self.pipeline if x[0] == self.tick]
        self.pipeline = [x for x in self.pipeline if x[0] != self.tick]
        for _, w, q in arrived:
            self.stock[w] += q
        delay_storm = "delay" in self.incident
        wrow = {}
        for w in WH:
            d = float(units[w])
            filled = min(self.stock[w], d)
            short = d - filled
            self.stock[w] -= filled
            short_tot += short
            dem_tot += d
            # update forecast (EWMA) and error
            e = d - self.rate[w]
            self.rate[w] += 0.15 * e
            self.err[w] = 0.9 * self.err[w] + 0.1 * abs(e)
            lt = int(self.rng.choice([3, 4, 6], p=[0.5, 0.3, 0.2])) + (3 if delay_storm and w in ("GCC-Dubai", "EU-Germany") else 0)
            target = self.rate[w] * 7 + 1.65 * 1.25 * self.err[w] * np.sqrt(7)   # cover lead time + review, plus SS
            pos = self.stock[w] + sum(q for _, ww, q in self.pipeline if ww == w)
            if pos < target:
                self.pipeline.append((self.tick + lt, w, target - pos))
            wrow[w] = {"stock": self.stock[w], "cover_ticks": self.stock[w] / max(self.rate[w], 1), "short": short}
            if short > 0 and self.rng.random() < 0.6:
                self._alert("crit", f"⛔ Stockout at {w}: {short:.0f} units could not ship")
            elif self.stock[w] / max(self.rate[w], 1) < 1.5 and self.rng.random() < 0.15:
                self._alert("warn", f"⚠️ {w} below 1.5 ticks of cover - replenishment on the way")

        # -------- reviews (noisy)
        n_rev = int(self.rng.poisson(2.2 * (2 if "spike" in self.incident else 1)))
        if n_rev:
            r = dg.make_reviews(n_rev, self.rng, ts - pd.Timedelta(seconds=3), ts,
                                neg_share=0.65 if delay_storm else 0.40,
                                stock_boost=0.25 if short_tot > 0 else 0.0,
                                notif_boost=0.45 if "blast" in self.incident else 0.0)
            if delay_storm:
                r.loc[r["topic"] == "delivery", "stars"] = r.loc[r["topic"] == "delivery", "stars"].clip(upper=2)
            ra = analyse(r)
            self.reviews = pd.concat([ra, self.reviews]).head(300)

        g = df.groupby("group")
        agg = {"tick": self.tick, "time": ts, "visitors": n, "orders": int(conv.sum()), "revenue": float(rev.sum()),
               "promo_cost": float(cost.sum()), "units": dem_tot, "short": short_tot,
               "neg_reviews": int((self.reviews.head(n_rev)["label"] == "Negative").sum()) if n_rev else 0,
               "reviews": n_rev}
        for name, gg in g:
            k = "smart" if name.startswith("Smart") else "blanket"
            agg[f"{k}_users"] = len(gg)
            agg[f"{k}_orders"] = int(gg["converted"].sum())
            agg[f"{k}_rev"] = float(gg["revenue"].sum())
            agg[f"{k}_cost"] = float(gg["promo_cost"].sum())
            agg[f"{k}_value"] = float(gg["value"].sum())
        for w in WH:
            agg[f"stock_{w}"] = wrow[w]["stock"]
            agg[f"cover_{w}"] = wrow[w]["cover_ticks"]
        self.ticks.append(agg)
        self.ticks = self.ticks[-400:]

    def warm_up(self, n: int = 40):
        for _ in range(n):
            self.step()

    # ------------------------------------------------------------------ views
    def tick_df(self):
        return pd.DataFrame(self.ticks)

    def visits_df(self, last: int = 60):
        return pd.concat(self.visits[-last:]) if self.visits else pd.DataFrame()
