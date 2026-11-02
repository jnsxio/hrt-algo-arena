# Methodology and experiment log

All numbers are annualized daily-PnL Sharpe ratios computed with the competition formula on walk-forward validation years: fit on all years before *Y*, score on *Y*. Selections use 2014 and 2015. 2016 is a follow-up check that had already been looked at in earlier stages, so it is not a pristine holdout. Raw outputs for every table are in `src/<stage>_results/`.

## 1. Data and metric

| Split | Years | Headlines | Headlines per day |
|---|---|---|---|
| Training (with returns) | 2013–2016 | 136,594 | 128–143 |
| Test (returns hidden) | 2017–2019 | 27,864 | 32–38 |

HRT's loader removes rows tagged `MOV` and headlines containing `HALT` or `RESUME`. Daily PnL is `sum(pred * return * weight)` over that day's headlines, with `weight = min(sqrt(adn), 50000)`. The score is `sqrt(252) * mean / std` of daily PnL.

Two consequences shaped everything else:

1. **Scale invariance.** A global multiplier on `pred` cancels in the Sharpe ratio. Only relative sizing across headlines and days matters.
2. **Density gap.** The test set has about 3.7× fewer headlines per day. With roughly independent bets, daily Sharpe scales with `sqrt(n)`, so validation Sharpes around 11 correspond to roughly 6 at test density. Later stages therefore also report a density-matched Sharpe, computed on random subsamples thinned to test density.

## 2. Baselines (`research_baselines.py`)

Selection on 2015, check on 2016.

| Model | 2015 | 2016 |
|---|---|---|
| Constant long | 0.57 | 1.01 |
| Fade past hour / past day | -0.73 / -0.81 | 0.78 / 1.03 |
| Ridge on past returns only | 0.53 | 0.82 |
| Ridge on TF-IDF 1–2-grams, α=100 | 7.29 | 7.51 |
| TF-IDF + past returns, α=100 | 7.05 | 7.51 |

The signal is in the text. Price-only mean reversion is weak in this data. Only 1.2–1.4% of validation headlines appear verbatim in the training years, so the text model is not memorizing repeated headlines.

## 3. Ridge tuning (`research_ridge_tuning.py`)

The grid covered n-gram range {1, 1–2, 1–3} × α {30, 100, 300, 1000}, and 1–3-grams with α=100 won on the 2014/2015 mean. Extensions on top of it:

| Variant | 2014 | 2015 | 2016 |
|---|---|---|---|
| Text only (1–3-grams, α=100) | 5.57 | 7.89 | 7.48 |
| Liquidity-weighted fit | 4.38 | 7.69 | 5.82 |
| Blend with day-fade, 0.1 | 5.73 | 7.96 | 7.78 |
| **Text × past-return interactions, 0.25** | 5.64 | **8.18** | **7.89** |
| Interactions, 1.0 | 3.86 | 7.48 | n/a |

Past returns enter normalized by volatility (floored at the training 5th percentile) and clipped to ±3. The interactions let a phrase carry a different sign or weight depending on the move that preceded it.

## 4. FinBERT and ElasticNet (`research_finbert_cache.py`, `research_finbert_elasticnet.py`)

FinBERT (`ProsusAI/finbert`, revision pinned in `finbert_results/model_source.json`) was run locally on every unique headline, with results cached in SQLite. The label order is read from the model config and asserted rather than assumed.

| Variant | 2014 | 2015 | 2016 |
|---|---|---|---|
| Ridge baseline (section 3) | 5.64 | 8.18 | 7.89 |
| Raw sentiment `p_pos - p_neg` alone | 2.85 | 6.41 | 4.54 |
| Calibrated sentiment alone | 2.46 | 6.30 | 5.29 |
| **Ridge + standardized probabilities × 0.1** | **5.95** | **9.23** | **7.92** |
| Ridge + probabilities × 1.0 | 5.83 | 9.32 | n/a |
| ElasticNet (L2=30, L1=1e-6) | 5.85 | 7.48 | 7.92 |
| 75% Ridge / 25% ElasticNet blend | mean 6.93 vs 6.91 | | |

FinBERT works as a small correction to the text model, not as a signal on its own. ElasticNet's L1 term only hurt, and the blend gain (+0.02) is noise, so the final model is Ridge + FinBERT × 0.1.

## 5. Prediction clipping and liquidity filters

Clipping (`research_prediction_clipping.py`): caps of 0.001–0.02 on `|pred|` were never better than no cap.

Liquidity filters (`research_liquidity_bootstrap.py`): trading only names above an `adn` threshold cut Sharpe sharply, e.g. from 5.64 to 3.06 at a 10k threshold in 2014. Illiquid names are where the signal is, which points directly at the sizing result below.

## 6. Exposure sizing (`research_exposure_sizing.py`, `research_sizing_refinement.py`)

The forecast is multiplied by `m = clip((med_vol / vol)^a * (weight / med_weight)^b, lower, upper)`, with medians and floors from training data only.

First pass, fixed rule shapes:

| Rule | 2014 | 2015 | 2016 |
|---|---|---|---|
| Unchanged | 5.95 | 9.23 | 7.92 |
| Inverse vol (a=1) | 6.26 | 8.75 | n/a |
| Inverse variance (a=2) | 5.72 | 7.52 | n/a |
| Favor lower liquidity (b=-0.5) | 7.83 | 10.55 | 8.76 |
| Favor higher liquidity (b=+0.5) | 4.26 | 6.85 | n/a |
| **Inverse vol × lower liquidity (a=1, b=-0.5)** | **9.05** | **11.12** | **9.88** |

Refinement over a ∈ {0.5, 1, 1.5, 2} × b ∈ {-0.25, -0.5, -0.75, -1} selected a=1, b=-1 with bounds [0.25, 4]: 2014/2015 mean 11.41. The return model was re-checked under the new objective, and α=100 with FinBERT × 0.1 stayed best.

**Why liquidity sizing works.** The evaluator weights PnL by `sqrt(adn)`, which concentrates risk in large caps. News moves small caps more, and less efficiently. With b ≈ -1.3, the effective exposure `m * weight` is roughly flat across liquidity and slightly tilted toward illiquid names, so each headline contributes comparable risk and the portfolio diversifies across many more effective bets.

## 7. Optuna sizing search (`research_optuna_sprint.py`, `research_optuna_extended.py`)

- **Search space:** a ∈ [0.25, 2], b ∈ [-1.75, -0.25], lower ∈ [0.05, 0.4] (log), upper ∈ [2, 10] (log).
- **Inputs:** frozen out-of-fold forecasts for 2014 and 2015; the return model is not refitted inside trials.
- **Objective:** `mean - 0.15 * std` of the density-matched Sharpe over 48 fixed thinning masks per year. Fixed masks make trials directly comparable.
- **Search:** TPE sampler, seed 77, 60 random start-up trials, 1,200 trials. The incumbent and the previous Optuna best are enqueued first.
- **Anti-overfitting check:** the top 12 trials were re-scored on 256 fresh masks per year, and the winner was picked there, not on the search masks.
- **Order of steps:** the selection was written to `selection.json` before 2016 was scored.

| Candidate | 2014 | 2015 | 2016 | 2016 density-matched |
|---|---|---|---|---|
| Incumbent (a=1, b=-1, [0.25, 4]) | 10.99 | 11.83 | 10.76 | 6.44 |
| Optuna extended (a=1.36, b=-1.32, [0.18, 9.45]) | 11.89 | 11.70 | 11.15 | 6.64 |
| Midpoint of the two | 11.57 | 11.86 | 11.08 | 6.62 |

In a paired 10-day block bootstrap on 2016, Optuna extended beats the incumbent in 89% of resamples (ΔSharpe +0.38, 95% CI [-0.24, 0.93]), and the midpoint does in 98% (+0.32, CI [0.00, 0.60]). Both were submitted: the midpoint first as the lower-variance step, then the full Optuna parameters.

A joint search over model and sizing parameters (`research_optuna_joint.py`, 80 trials) landed on essentially the same sizing, with α≈160, and is not among the recorded submissions.

## 8. Leaderboard log

| Order | File | Score |
|---|---|---|
| 1 | `predictions_ridge_finbert.csv` | 2.701 |
| 2 | `predictions_ridge_finbert_sized.csv` | 4.310 |
| 3 | `predictions_ridge_finbert_sized_v2.csv` | 4.838 |
| 4 | `predictions_optuna_midpoint.csv` | 5.000 |
| 5 | `predictions_optuna_extended.csv` | **5.052** (final) |

Every step that improved validation also improved the leaderboard, in the same order. The validation-to-test gap matches the density argument in section 1: for submission 3, the density-matched 2016 Sharpe over 200 thinned subsamples has mean 6.5 and range 4.0–9.2, and the test score was 4.84.

## 9. Script map

| Script | Role |
|---|---|
| `research_baselines.py` | Constant, fade and first Ridge baselines; defines the shared `metrics()` |
| `research_ridge_tuning.py` | n-gram / α grid, past-return blends and interactions |
| `research_liquidity_bootstrap.py` | Liquidity filters; defines the shared paired block `bootstrap()` |
| `research_prediction_clipping.py` | Prediction caps |
| `research_finbert_cache.py` | Local FinBERT download and resumable inference cache |
| `research_finbert_elasticnet.py` | FinBERT feature variants and ElasticNet; saves out-of-fold predictions |
| `research_elasticnet_blend.py` | Ridge/ElasticNet convex blend |
| `research_exposure_sizing.py` | First sizing rules; defines `multiplier()` |
| `research_sizing_refinement.py` | Sizing grid, model re-check, density stress test |
| `research_optuna_sprint.py` | First Optuna sizing search; defines `prepare()`, `predictions()`, `evaluate()` |
| `research_optuna_extended.py` | 1,200-trial search with fresh-mask verification; writes the final CSV |
| `research_optuna_joint.py` | Joint model + sizing search |
| `make_ridge_finbert_csv.py` | Refit the final model on 2013–2016; asserts it reproduces research predictions (< 1e-8) |
| `make_sized_csv.py`, `make_refined_sized_csv.py` | Apply frozen sizing to the base forecast |
| `make_optuna_line_probe.py` | Interpolate or extrapolate between sizing candidates (probe, not among the recorded submissions) |
