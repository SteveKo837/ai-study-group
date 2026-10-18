# One-clerk model-specific strategy benchmark

## Purpose

This experiment isolates one clerk and lets each model use a suitable inference precision. It also tests whether `WAIT` should be removed whenever useful work exists.

## Executive result

**Laya wins when its real latency advantage is represented.** Its best strategy, `laya-0.1s-no-wait`, produced NT$1281 revenue, served 12.90, and abandoned 2.80 customers per episode.
Against Gemma's best strategy (`gemma-0.5s-no-wait`), Laya generated 7.2% more revenue, served 4.0% more customers, and reduced abandonment by 15.2%.
Against the rule baseline, best-strategy Laya generated 2.8% more revenue and reduced abandonment by 3.4%.
Across matched seeds, Laya beat Gemma on revenue in 6/10 episodes with 2 ties, and had fewer abandonments in 6/10 with 3 ties.

## Method

- 9 strategies × 10 matched seeds; 300 simulated seconds per episode; no speed-up (1× during inference).
- Customer limit 15; demand 6912 customers/day equivalent.
- Arrival draws occur once per simulated second, so all tick precisions receive the same seeded demand schedule.
- Customer patience, movement, browsing, item pickup, clerk travel, and service work use elapsed simulated time rather than step counts.
- Model inference runs synchronously on the process main thread; after each measured call, the store advances by that elapsed time before the decision is validated. This event-driven method preserves customer waiting while avoiding the unstable PyTorch-MPS background-thread path.
- Warm-up/model-load time is excluded from episode metrics.
- Runtime devices: Laya=mps; Gemma=llama-server.

## Results

| Strategy | Revenue | Served | Abandoned | Checkout wait | Service completion | Mean latency | Mean model delay | Applied | Invalid | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rule-baseline | NT$1246 | 12.70 | 2.90 | 20.9s | 34.3% | 0.0ms | 0.00s | 100.0% | 0.0% | 1.000 |
| gemma-1s-wait | NT$1210 | 12.30 | 3.60 | 24.8s | 39.1% | 465.9ms | 1.00s | 83.3% | 16.7% | 0.450 |
| gemma-1s-no-wait | NT$1151 | 12.30 | 3.70 | 21.4s | 32.8% | 462.9ms | 1.00s | 100.0% | 0.0% | 0.200 |
| gemma-0.5s-no-wait | NT$1195 | 12.40 | 3.30 | 22.0s | 34.5% | 464.2ms | 0.63s | 100.0% | 0.0% | 0.765 |
| laya-1s-wait | NT$1164 | 12.20 | 3.90 | 17.1s | 39.7% | 59.0ms | 1.00s | 30.8% | 69.2% | 0.444 |
| laya-1s-no-wait | NT$1168 | 12.00 | 3.50 | 18.8s | 31.8% | 60.4ms | 1.00s | 100.0% | 0.0% | 0.177 |
| laya-0.5s-no-wait | NT$1187 | 12.40 | 3.60 | 20.5s | 32.3% | 59.9ms | 0.50s | 100.0% | 0.0% | 0.196 |
| laya-0.25s-no-wait | NT$1265 | 12.70 | 3.00 | 19.9s | 36.1% | 59.2ms | 0.25s | 100.0% | 0.0% | 0.633 |
| laya-0.1s-no-wait | NT$1281 | 12.90 | 2.80 | 18.2s | 33.3% | 63.3ms | 0.12s | 100.0% | 0.0% | 0.775 |

The quality score is an equal-weight normalization within each model over revenue, served, abandonment, checkout wait, and service completion. It selects a model-specific operating strategy; it is not a universal business utility function.

## Best strategy per model

- **Gemma: `gemma-0.5s-no-wait`** — NT$1195 revenue, 12.40 served, 3.30 abandoned, 100.0% decisions applied.
- **Laya: `laya-0.1s-no-wait`** — NT$1281 revenue, 12.90 served, 2.80 abandoned, 100.0% decisions applied.

## Interpretation guardrails

- Compare a model's filtered and unfiltered 1-second strategies to isolate the `WAIT` choice effect.
- Compare only filtered strategies across tick sizes to isolate timing precision.
- Sub-second precision can expose latency differences, but policy quality can still dominate operational outcomes.
- In this run Laya averaged roughly 63ms at its best setting versus 464ms for Gemma. At a one-second tick that difference is hidden; at 0.1s/0.5s it changes clerk availability.
- Laya's timing curve is not guaranteed to be monotonic because different completion boundaries can change subsequent store states; use the matched-seed aggregate rather than assuming a smaller tick always wins.
- 10 seeds provide an engineering comparison, not a production forecast.
