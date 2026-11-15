# 📈 Temu Growth Lab

A live, colourful analytics dashboard for our **SBR group case study on Temu**. It works through three business
problems with the nine *Data-Driven Growth with Python* techniques (Barış Karaman, Towards Data Science),
plus a few supply-chain tools.

> **All data is simulated.** This is a student project. It is not affiliated with Temu or PDD Holdings. The synthetic
> data is calibrated to the assumptions in our Minutes of Meeting: 40% of new buyers order again within 90 days,
> promotions cost 15% of gross sales, about 1 in 3 coupon orders is truly incremental, and 12% of local-warehouse
> orders hit a stockout or delay.

## What's inside

| Tab | What it shows | Blogs / tools |
|---|---|---|
| 🔴 Live | Simulated shoppers stream in every few seconds. Half get the old blanket 20% coupon and half get **smart targeting** (uplift model + cost optimisation), as a live A/B test. Orders draw down six local warehouses, which are replenished with forecast + safety stock. Noisy multilingual reviews (typos, emojis, sarcasm, spam) are scored as they arrive. You can trigger a sale spike, a delivery-delay storm or a notification blast. | 7, 8, 9, 6 |
| 🏆 Findings | Before → predicted-after for every problem, compared with the targets in the minutes | all |
| 📏 Metrics | North Star metric, revenue, active customers, retention, cohort heatmap, 90-day repeat rate | 1 |
| 🔁 P1 Retention | RFM + K-means, CLV and LTV:CAC by channel, churn model, next-purchase-day model, retention plan projection and A/B design | 1, 2, 3, 4, 5, 9 |
| 🎟️ P2 Coupons | 4-arm randomised experiment, market response, uplift modelling (class transformation + T-learner), the 4 segments, cost optimisation with an order guardrail, 4-week stratified A/B test | 7, 8, 9 |
| 📦 P3 Warehouses | LSTM vs tuned XGBoost vs baselines, inventory simulation, service-level frontier, ABC-XYZ, safety-stock calculator, stratified pilot test | 6, 5, 2, 9 + ABC-XYZ, safety stock |
| 💬 Reviews | Sentiment and topic analysis of 4,000 noisy reviews | — |
| 🧪 Model lab | Default vs tuned cross-validated scores for every model (RandomizedSearchCV), best hyperparameters, held-out test results | — |
| 📐 Formulas | Every formula used, grouped by blog | — |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the data and models (optional, ~30 minutes on 2 CPUs)

```bash
pip install -r requirements-train.txt
python train.py
```

`train.py` generates the synthetic data (`data/`), runs every analysis, tunes 4 model families per task with
cross-validation, trains the LSTM, and writes everything the dashboard reads to `artifacts/` and `models/`.

## Project layout

```
app.py                 Streamlit dashboard
train.py               data → analyses → tuned models → artifacts
src/data_gen.py        synthetic customers, orders, coupon experiment, warehouse demand, SKUs, noisy reviews
src/p1_retention.py    Blogs 1-5 + retention projection
src/p2_coupons.py      Blogs 7-9: market response, uplift, cost optimisation, A/B test
src/p3_inventory.py    Blog 6 forecasting (XGBoost + LSTM), inventory simulation, ABC-XYZ, pilot test
src/tuning.py          default-vs-tuned model comparison
src/stats.py           z-test, Welch t-test, sample size, stratified split
src/reviews.py         spam filter, lexicon sentiment with negation and sarcasm, topic tagging
src/live.py            the live stream simulator
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app**.
3. Pick the repository and branch, set **Main file path** to `app.py`, and click **Deploy**.

## Honest caveats

- Model accuracy reflects the randomness built into the simulation. Compare each model with its baseline, as the Model lab does.
- The Problem 1 projection uses assumed action effects. You can change them with the sidebar sliders.
- Predicted improvements are a direction and a scale, not a promise. Real Temu data would be needed to confirm them.
