"""
Synthetic data generator for the Temu growth-analytics case study.

Everything here is SIMULATED for a classroom project. No real Temu data is used.
The generator is calibrated so the "today" baseline matches the assumptions in
the group's Minutes of Meeting:
  * ~40% of new buyers place a 2nd order within 90 days (60% do not)
  * promotions cost ~15% of gross sales
  * only ~1 in 3 coupon orders is truly incremental
  * ~12% of local-warehouse orders hit a stockout or delay
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-10-31")  # analysis "as of" date

# region -> (warehouse, share of customers, price level multiplier)
REGIONS = {
    "United States": ("US-Texas", 0.30, 1.00),
    "United Kingdom": ("UK-Midlands", 0.10, 0.95),
    "Germany": ("EU-Germany", 0.12, 1.00),
    "France": ("EU-Germany", 0.10, 0.98),
    "UAE": ("GCC-Dubai", 0.10, 1.15),
    "Saudi Arabia": ("GCC-Dubai", 0.08, 1.10),
    "Mexico": ("LATAM-Mexico", 0.10, 0.80),
    "Brazil": ("LATAM-Brazil", 0.10, 0.78),
}
WAREHOUSES = ["US-Texas", "UK-Midlands", "EU-Germany", "GCC-Dubai", "LATAM-Mexico", "LATAM-Brazil"]

# channel -> (share, customer acquisition cost in USD, loyalty shift)
CHANNELS = {
    "TikTok": (0.22, 9.0, -0.02),
    "Meta": (0.20, 11.0, 0.00),
    "Google Search": (0.14, 14.0, 0.08),
    "Referral game": (0.16, 6.5, -0.10),
    "Influencer": (0.10, 12.5, -0.03),
    "TV / Big events": (0.06, 19.0, -0.08),
    "Organic": (0.12, 1.5, 0.10),
}

# category -> (share of orders, basket value multiplier, return-rate add-on)
CATEGORIES = {
    "Home & Kitchen": (0.20, 1.00, 0.00),
    "Fashion": (0.18, 0.95, 0.06),
    "Phone & Electronics Acc.": (0.13, 1.20, 0.02),
    "Beauty": (0.10, 0.80, 0.01),
    "Toys & Games": (0.08, 1.05, 0.01),
    "Sports & Outdoors": (0.07, 1.25, 0.02),
    "Pet Supplies": (0.06, 0.90, 0.00),
    "Automotive": (0.06, 1.30, 0.01),
    "Office & School": (0.06, 0.75, 0.00),
    "Garden": (0.06, 1.10, 0.01),
}

GROSS_MARGIN = 0.25  # contribution margin before promotions


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pick(rng, d: dict, n: int, idx: int = 0):
    keys = list(d.keys())
    p = np.array([v[idx] if isinstance(v, tuple) else v for v in d.values()], dtype=float)
    p = p / p.sum()
    return rng.choice(keys, size=n, p=p)


# --------------------------------------------------------------------------
# 1) Customers + orders  (Problem 1, Blogs 1-5)
# --------------------------------------------------------------------------
def make_customers_orders(n_customers: int = 40_000, seed: int = SEED,
                          first_churn_shift: float = 0.0):
    rng = np.random.default_rng(seed)
    n_days = (END - START).days
    w = np.linspace(1.0, 2.2, n_days + 1)  # Temu kept growing
    signup_day = rng.choice(np.arange(n_days + 1), size=n_customers, p=w / w.sum())
    region = _pick(rng, REGIONS, n_customers, idx=1)
    channel = _pick(rng, CHANNELS, n_customers, idx=0)

    ch_shift = np.array([CHANNELS[c][2] for c in channel])
    loyalty = np.clip(rng.beta(2.0, 3.0, n_customers) + ch_shift, 0.01, 0.99)
    price_sens = np.clip(rng.beta(2.5, 2.5, n_customers)
                         + np.isin(region, ["Mexico", "Brazil"]) * 0.10, 0.01, 0.99)
    cac = np.array([CHANNELS[c][1] for c in channel]) * rng.lognormal(0, 0.18, n_customers)

    cust = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n_customers)],
        "signup_date": START + pd.to_timedelta(signup_day, unit="D"),
        "region": region,
        "warehouse": [REGIONS[r][0] for r in region],
        "channel": channel,
        "cac_usd": cac.round(2),
        "_loyalty": loyalty,
        "_price_sens": price_sens,
    })

    cat_keys = list(CATEGORIES)
    cat_p = np.array([v[0] for v in CATEGORIES.values()])
    cat_p = cat_p / cat_p.sum()

    rows = []
    end_ord = END.toordinal()
    for i, r in enumerate(cust.itertuples(index=False)):
        lo, ps = loyalty[i], price_sens[i]
        reg_mult = REGIONS[r.region][2]
        mean_gap = 12 + 75 * (1 - lo)
        aov_base = 20 * reg_mult * (0.8 + 0.6 * lo)
        fav = rng.dirichlet(cat_p * 8)
        coupon_p = 0.30 + 0.50 * ps
        q_first = _sigmoid(1.50 - 3.6 * lo + first_churn_shift)  # leaves after 1st order
        q_next = 0.06 + 0.32 * (1 - lo)
        d = r.signup_date
        k = 0
        while d.toordinal() <= end_ord:
            cat = cat_keys[rng.choice(len(cat_keys), p=fav)]
            value = rng.lognormal(np.log(aov_base * CATEGORIES[cat][1]), 0.50)
            coupon = rng.random() < coupon_p
            discount = value * rng.uniform(0.18, 0.36) if coupon else 0.0
            peak = (d.month in (11, 12)) or (r.warehouse == "GCC-Dubai" and d.month in (2, 3))
            delayed = rng.random() < (0.11 + 0.07 * peak)
            delivery = int(rng.integers(3, 7) + (rng.integers(5, 13) if delayed else 0))
            returned = rng.random() < (0.05 + CATEGORIES[cat][2])
            rows.append((r.customer_id, d, cat, r.region, r.warehouse, round(value, 2),
                         bool(coupon), round(discount, 2), delivery, bool(delayed), bool(returned), k))
            q = (q_first if k == 0 else q_next) + 0.14 * delayed + 0.08 * returned
            if rng.random() < q:
                break
            gap = rng.gamma(shape=3.5, scale=mean_gap / 3.5)
            d = d + pd.Timedelta(days=max(1, int(round(gap))))
            k += 1

    orders = pd.DataFrame(rows, columns=[
        "customer_id", "order_date", "category", "region", "warehouse", "gross_value",
        "coupon_used", "discount", "delivery_days", "delayed", "returned", "order_seq"])
    orders["net_revenue"] = (orders["gross_value"] - orders["discount"]).round(2)
    orders.insert(0, "order_id", [f"O{i:07d}" for i in range(len(orders))])
    return cust, orders


# --------------------------------------------------------------------------
# 2) Coupon experiment  (Problem 2, Blogs 7, 8, 9)
# --------------------------------------------------------------------------
ARMS = ["No offer", "20% off", "Buy-one-get-one", "Free shipping"]
OFFER_COST = {  # cost to Temu when a converted user redeems
    "No offer": lambda rev: 0.0 * rev,
    "20% off": lambda rev: 0.20 * rev,
    "Buy-one-get-one": lambda rev: 0.30 * rev,
    "Free shipping": lambda rev: 0.0 * rev + 2.8,
}


def campaign_truth(df: pd.DataFrame):
    """True conversion probability for every arm (known only because data is simulated)."""
    lo, ps, an, bk = df["_loyalty"], df["_price_sens"], df["_annoy"], df["_basket"]
    notif = df["notifications_per_day"]
    annoy_eff = an * np.clip(notif / 5.0, 0, 1.6)
    base = -3.25 + 3.4 * lo + 0.25 * (df["sessions_30d"] > 10) - 0.4 * annoy_eff
    p = {"No offer": _sigmoid(base)}
    contact_penalty = -2.8 * np.clip(annoy_eff - 0.22, 0, None)
    e20 = 3.5 * ps ** 2.6 * (1 - lo) ** 0.8 + contact_penalty
    ebogo = 3.2 * (bk * ps) ** 1.6 * (1 - lo) + contact_penalty
    efs = 1.1 * (df["avg_order_value"] < 18) * ps * (1 - lo) + 0.2 * ps + contact_penalty
    p["20% off"] = _sigmoid(base + e20)
    p["Buy-one-get-one"] = _sigmoid(base + ebogo)
    p["Free shipping"] = _sigmoid(base + efs)
    return pd.DataFrame(p)


def true_segment(p: pd.DataFrame) -> pd.Series:
    up = p["20% off"] - p["No offer"]
    seg = np.where(up < -0.01, "Sleeping dog",
          np.where(up >= 0.05, "Persuadable",
          np.where(p["No offer"] >= 0.15, "Sure thing", "Lost cause")))
    return pd.Series(seg, index=p.index)


def make_campaign(n: int = 80_000, seed: int = SEED + 1, assign: bool = True,
                  arm_probs=(0.25, 0.25, 0.25, 0.25)):
    rng = np.random.default_rng(seed)
    loyalty = rng.beta(1.6, 3.4, n)
    price_sens = rng.beta(2.4, 2.4, n)
    annoy = rng.beta(1.4, 4.0, n)
    basket = rng.beta(2.0, 3.0, n)
    region = _pick(rng, REGIONS, n, idx=1)
    channel = _pick(rng, CHANNELS, n, idx=0)

    orders_12m = rng.poisson(0.4 + 16 * loyalty ** 1.6)
    aov = rng.lognormal(np.log(14 + 22 * (1 - price_sens) + 6 * basket), 0.30)
    df = pd.DataFrame({
        "user_id": [f"U{i:06d}" for i in range(n)],
        "region": region,
        "channel": channel,
        "device": rng.choice(["Android", "iOS", "Web"], n, p=[0.52, 0.38, 0.10]),
        "recency_days": np.clip(rng.exponential(8 + 150 * (1 - loyalty) ** 1.5), 0, 365).round(),
        "orders_12m": orders_12m,
        "avg_order_value": aov.round(2),
        "spend_12m": (orders_12m * aov).round(2),
        "coupon_use_rate": np.clip(0.15 + 0.7 * price_sens + rng.normal(0, 0.10, n), 0, 1).round(3),
        "cart_abandon_rate": np.clip(0.15 + 0.55 * price_sens + rng.normal(0, 0.12, n), 0, 1).round(3),
        "sessions_30d": rng.poisson(1 + 26 * loyalty),
        "items_per_order": (1 + rng.poisson(3.2 * basket, n)),
        "notifications_per_day": np.clip(rng.gamma(2.0, 1.6, n), 0.2, 12).round(1),
        "push_open_rate": np.clip(0.42 - 0.45 * annoy + rng.normal(0, 0.06, n), 0, 1).round(3),
        "muted_notifications": (rng.random(n) < 0.05 + 0.6 * annoy ** 2).astype(int),
        "uninstalls_12m": rng.poisson(0.05 + 1.4 * annoy ** 2),
        "is_referral": (channel == "Referral game").astype(int),
        "_loyalty": loyalty, "_price_sens": price_sens, "_annoy": annoy, "_basket": basket,
    })
    truth = campaign_truth(df)
    for a in ARMS:
        df[f"_p_{a}"] = truth[a].values
    df["_segment"] = true_segment(truth).values
    if assign:
        arm = rng.choice(ARMS, size=n, p=list(arm_probs))
        df["offer"] = arm
        p_arm = truth.values[np.arange(n), [ARMS.index(a) for a in arm]]
        conv = rng.random(n) < p_arm
        basket_mult = np.where(arm == "Buy-one-get-one", 1.25, 1.0)
        rev = np.where(conv, df["avg_order_value"] * basket_mult * rng.lognormal(0, 0.25, n), 0.0)
        df["conversion"] = conv.astype(int)
        df["revenue"] = rev.round(2)
        df["promo_cost"] = np.array([OFFER_COST[a](r) if c else 0.0
                                     for a, r, c in zip(arm, rev, conv)]).round(2)
    return df


# --------------------------------------------------------------------------
# 3) Warehouse demand  (Problem 3, Blogs 5, 6, 9 + ABC-XYZ, safety stock)
# --------------------------------------------------------------------------
RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19")]


def event_flags(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=weeks)
    wk = weeks.isocalendar().week.values.astype(int)
    f["black_friday"] = ((weeks.month == 11) & (weeks.day >= 20)).astype(int)
    f["christmas"] = ((weeks.month == 12) & (weeks.day <= 21)).astype(int)
    f["back_to_school"] = ((weeks.month == 8) | ((weeks.month == 9) & (weeks.day <= 7))).astype(int)
    f["singles_day"] = ((weeks.month == 11) & (weeks.day >= 7) & (weeks.day <= 14)).astype(int)
    f["summer_sale"] = ((weeks.month == 7) & (weeks.day >= 7) & (weeks.day <= 21)).astype(int)
    ram = np.zeros(len(weeks), dtype=int)
    for a, b in RAMADAN:
        ram |= ((weeks >= pd.Timestamp(a) - pd.Timedelta(days=6)) & (weeks <= pd.Timestamp(b))).astype(int)
    f["ramadan"] = ram
    f["week_of_year"] = wk
    return f


def make_weekly_demand(seed: int = SEED + 2):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-11-06", "2026-10-26", freq="W-MON")
    ev = event_flags(weeks)
    t = np.arange(len(weeks))
    rows = []
    wh_level = {"US-Texas": 5200, "UK-Midlands": 1500, "EU-Germany": 2600,
                "GCC-Dubai": 1400, "LATAM-Mexico": 1500, "LATAM-Brazil": 1300}
    for wh in WAREHOUSES:
        # local share ramps after the de minimis changes (US May-2025, EU mid-2026, others gradual)
        if wh == "US-Texas":
            shift = _sigmoid((weeks - pd.Timestamp("2025-05-05")).days / 21)
        elif wh in ("EU-Germany", "UK-Midlands"):
            shift = _sigmoid((weeks - pd.Timestamp("2026-06-01")).days / 28)
        else:
            shift = _sigmoid((weeks - pd.Timestamp("2025-10-01")).days / 60)
        for cat, (share, _, _) in CATEGORIES.items():
            lvl = wh_level[wh] * share * rng.uniform(0.8, 1.2)
            season = 1 + 0.10 * np.sin(2 * np.pi * (t / 52.18) + rng.uniform(0, 2 * np.pi))
            evm = (1 + 0.85 * ev["black_friday"] + 0.45 * ev["christmas"] + 0.25 * ev["singles_day"]
                   + 0.15 * ev["summer_sale"]
                   + (0.45 if cat in ("Office & School", "Fashion") else 0.10) * ev["back_to_school"]
                   + (0.65 if wh == "GCC-Dubai" else 0.0) * ev["ramadan"]
                   + (0.55 if wh == "GCC-Dubai" else 0.0) * ev["black_friday"])  # White Friday
            trend = 1 + 0.004 * t
            local = 0.25 + 0.75 * shift
            mu = lvl * season * evm.values * trend * local
            y = rng.poisson(np.maximum(mu * rng.lognormal(0, 0.08, len(t)), 1))
            for wk_, yy in zip(weeks, y):
                rows.append((wk_, wh, cat, int(yy)))
    dem = pd.DataFrame(rows, columns=["week", "warehouse", "category", "units"])
    return dem, ev


def make_skus(n_sku: int = 240, seed: int = SEED + 3, n_weeks: int = 78):
    """SKU-level weekly demand (last 78 weeks) for ABC-XYZ and inventory simulation."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(end="2026-10-26", periods=n_weeks, freq="W-MON")
    ev = event_flags(weeks)
    cats = list(CATEGORIES)
    out, meta = [], []
    pareto = rng.pareto(1.25, n_sku) + 1
    for s in range(n_sku):
        cat = cats[s % len(cats)]
        mean = 6 + 260 * pareto[s] / pareto.max() * rng.uniform(0.6, 1.4)
        cv_type = rng.choice(["steady", "seasonal", "lumpy"], p=[0.45, 0.30, 0.25])
        price = round(float(rng.lognormal(np.log(9), 0.6)), 2)
        if cv_type == "steady":
            mu = mean * (1 + rng.normal(0, 0.05, n_weeks))
        elif cv_type == "seasonal":
            mu = mean * (1 + 0.55 * np.sin(2 * np.pi * np.arange(n_weeks) / 52.18 + rng.uniform(0, 6))
                         + 1.0 * ev["black_friday"].values)
        else:
            mu = mean * rng.choice([0, 0.3, 1, 3.5], n_weeks, p=[0.35, 0.25, 0.25, 0.15])
        y = rng.poisson(np.maximum(mu, 0.05))
        meta.append((f"SKU{s:04d}", cat, price, cv_type))
        for w, v in zip(weeks, y):
            out.append((f"SKU{s:04d}", w, int(v)))
    sku = pd.DataFrame(meta, columns=["sku", "category", "unit_price", "_pattern"])
    sd = pd.DataFrame(out, columns=["sku", "week", "units"])
    return sku, sd


# --------------------------------------------------------------------------
# 4) Customer reviews with realistic noise
# --------------------------------------------------------------------------
_TOPICS = {
    "delivery": {
        "pos": ["arrived in {d} days, faster than expected", "delivery was quick and tracking worked",
                "local warehouse shipping is so much faster now"],
        "neg": ["took {d} days to arrive, way too slow", "package delayed again, {d} days and counting",
                "tracking stuck for a week, still waiting", "said 5 days, came in {d}"],
    },
    "quality": {
        "pos": ["good quality for the price", "exactly like the photos", "sturdy and works well, surprised tbh"],
        "neg": ["cheap material, broke after 2 days", "looks nothing like the picture",
                "size was completely wrong", "stopped working after one use"],
    },
    "price_coupon": {
        "pos": ["used a coupon and got it super cheap", "free shipping deal made it worth it",
                "BOGO offer was great, got 2 for 1"],
        "neg": ["coupon would not apply at checkout", "the 'free gift' needed 5 referrals, not free at all",
                "price went up after tariffs", "spin the wheel prize expired before I could use it"],
    },
    "stock": {
        "pos": ["item was in stock locally, got it fast"],
        "neg": ["out of stock after I paid", "order cancelled, item unavailable",
                "back-ordered for weeks", "only 1 of 3 items shipped, rest out of stock"],
    },
    "app_notifications": {
        "pos": ["app is fun and easy to use"],
        "neg": ["too many notifications, uninstalling", "spam pop-ups every 5 minutes",
                "stop sending me push messages!!", "the games and pop-ups are annoying"],
    },
    "service": {
        "pos": ["support refunded me quickly", "return process was easy"],
        "neg": ["support never replied", "refund still pending after 3 weeks", "return label never came"],
    },
}
_NOISE_LANG = ["muy bueno", "não chegou ainda", "sehr schlecht", "ممتاز", "pas mal", "شحن بطيء"]
_EMOJI_POS = ["😍", "👍", "🔥", "✨", "😊"]
_EMOJI_NEG = ["😡", "👎", "🙄", "😤", "💀"]
_SPAM = ["CLICK HERE to win $500 Temu credit!!! bit.ly/xx", "follow me for free gifts code 1234",
         "best deals at my store visit now", "aaaaaaa", "."]


def _typo(rng, s: str) -> str:
    if len(s) < 6:
        return s
    i = int(rng.integers(1, len(s) - 2))
    op = rng.integers(0, 3)
    if op == 0:
        return s[:i] + s[i + 1:]
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def make_reviews(n: int, rng, start: pd.Timestamp, end: pd.Timestamp,
                 neg_share: float = 0.42, stock_boost: float = 0.0, notif_boost: float = 0.0):
    topics = list(_TOPICS)
    tw = np.array([0.24, 0.24, 0.18, 0.12 + stock_boost, 0.10 + notif_boost, 0.12])
    tw = tw / tw.sum()
    span = max(1, int((end - start).total_seconds()))
    out = []
    for _ in range(n):
        ts = start + pd.Timedelta(seconds=int(rng.integers(0, span)))
        region = rng.choice(list(REGIONS))
        if rng.random() < 0.04:
            out.append((ts, region, "spam", int(rng.integers(1, 6)), str(rng.choice(_SPAM)), 1))
            continue
        topic = rng.choice(topics, p=tw)
        neg = rng.random() < neg_share
        text = str(rng.choice(_TOPICS[topic]["neg" if neg else "pos"])).format(d=int(rng.integers(9, 26)))
        stars = int(rng.choice([1, 2], p=[0.6, 0.4])) if neg else int(rng.choice([4, 5], p=[0.35, 0.65]))
        if rng.random() < 0.08:  # sarcasm: positive words, negative meaning
            text = f"great, {text} 🙄" if neg else text
        if rng.random() < 0.06:  # star rating does not match the text
            stars = 6 - stars
        if rng.random() < 0.25:
            text = _typo(rng, text)
        if rng.random() < 0.15:
            text = text.upper()
        if rng.random() < 0.30:
            text += " " + str(rng.choice(_EMOJI_NEG if neg else _EMOJI_POS))
        if rng.random() < 0.10:
            text += " " + str(rng.choice(_NOISE_LANG))
        if rng.random() < 0.05:
            text = str(rng.choice(["ok", "meh", "fine i guess", "?"]))
            stars = 3
        out.append((ts, region, topic, stars, text, 0))
    return pd.DataFrame(out, columns=["timestamp", "region", "topic", "stars", "text", "is_spam"]).sort_values("timestamp")


if __name__ == "__main__":
    import time
    t0 = time.time()
    c, o = make_customers_orders()
    print(len(c), len(o), round(time.time() - t0, 1), "s")
