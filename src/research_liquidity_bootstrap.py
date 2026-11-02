"""Liquidity ablation and paired block bootstrap; no OOS reads or submissions.

Uses the previously selected words13, alpha=100, interaction strength=.25.
Threshold choices use only expanding-window 2014/2015 scores. 2016 has already
been inspected in previous research and is explicitly a reused check.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import news
from research_baselines import metrics
from research_ridge_tuning import design, fit, mr_features, interactions

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'liquidity_bootstrap_results'
CUTOFFS = [0, 2500, 5000, 10000]


def forecasts(train, valid):
    a, b = design(train, valid, 'words13')
    _, text = fit(a, b, train, 100.)
    zt, zv = mr_features(train, valid)
    _, with_returns = fit(interactions(a, zt, .25), interactions(b, zv, .25), train, 100.)
    return {'text': text, 'with_returns': with_returns}


def bootstrap(frame, baseline, repetitions=5000):
    """Paired circular block bootstrap of full daily portfolios, fixed fitted models.

    Includes zero-position dates. Same resampled dates for every strategy.
    Reports conditional uncertainty, not selection-adjusted significance.
    """
    x = frame.to_numpy()
    n = len(x)
    base = list(frame.columns).index(baseline)
    estimates = np.sqrt(252) * x.mean(axis=0) / (x.std(axis=0, ddof=1) + 1e-12)
    rows = []
    for length in [5, 10, 20]:
        rng = np.random.default_rng(42)
        samples = []
        # Batches keep resampled arrays small.
        for start in range(0, repetitions, 100):
            count = min(100, repetitions - start)
            starts = rng.integers(0, n, size=(count, int(np.ceil(n / length))))
            indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(count, -1)[:, :n]
            resampled = x[indices]
            samples.append(np.sqrt(252) * resampled.mean(axis=1) /
                           (resampled.std(axis=1, ddof=1) + 1e-12))
        values = np.concatenate(samples)
        for j, name in enumerate(frame.columns):
            delta = values[:, j] - values[:, base]
            ci = np.quantile(values[:, j], [.025, .975])
            dci = np.quantile(delta, [.025, .975])
            rows.append({'model': name, 'baseline': baseline, 'block_days': length,
                         'replicates': repetitions, 'sharpe': estimates[j],
                         'sharpe_ci_low': ci[0], 'sharpe_ci_high': ci[1],
                         'delta_sharpe': estimates[j] - estimates[base],
                         'delta_ci_low': dci[0], 'delta_ci_high': dci[1],
                         'resample_fraction_delta_positive': float(np.mean(delta > 0))})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None,
                            label='liquidity research').get_data()
    years = df.index.get_level_values('date').year
    records, daily, coverage = [], {}, []
    selected = None
    for year in [2014, 2015, 2016]:
        train, valid = df[years < year], df[years == year]
        all_predictions = forecasts(train, valid)
        def record(name, pred, ntrain):
            score, pnl = metrics(valid, pred)
            records.append({'year': year, 'model': name, 'training_rows': ntrain,
                            'active_fraction': float(np.mean(pred != 0)), **score})
            daily[(year, name)] = pnl
            print(f'{year} {name}: Sharpe={score["sharpe"]:.4f}', flush=True)
        for cutoff in CUTOFFS:
            keep = valid.weight.to_numpy() >= cutoff
            coverage.append({'year': year, 'cutoff': cutoff,
                             'headline_fraction': float(keep.mean()),
                             'weight_fraction': float(valid.weight[keep].sum() / valid.weight.sum())})
            # All configurations are prespecified; 2016 is diagnostic, never selects.
            for family, pred in all_predictions.items():
                record(f'{family}_trade_ge_{cutoff}', pred * keep, len(train))
            if cutoff:
                subset = train[train.weight >= cutoff]
                for family, pred in forecasts(subset, valid).items():
                    record(f'{family}_train_and_trade_ge_{cutoff}', pred * keep, len(subset))
        for family, pred in all_predictions.items():
            record(f'{family}_price_ge_5', pred * (valid.prev_close.to_numpy() >= 5), len(train))
        if year == 2015:
            table = pd.DataFrame(records)
            means = table.groupby('model').sharpe.mean().sort_values(ascending=False)
            selected = {'best_model': means.index[0],
                        'best_liquidity_cutoff_with_returns': means[
                            means.index.str.startswith('with_returns_trade_ge_')].index[0],
                        'selection_years': [2014, 2015], 'check_year': 2016,
                        'check_previously_seen': True, 'cutoffs': CUTOFFS,
                        'oos_accessed': False, 'submission_written': False,
                        'selection_mean_sharpes': means.to_dict()}
            (OUT/'selection.json').write_text(json.dumps(selected, indent=2))
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv', index=False)
    # Store year-specific daily series without mixing train/test years.
    pd.DataFrame({f'{y}_{name}': v for (y, name), v in daily.items()}).to_csv(OUT/'daily_pnl.csv')
    check = pd.DataFrame({name: pnl for (year, name), pnl in daily.items() if year == 2016})
    rows = bootstrap(check, 'with_returns_trade_ge_0')
    # Separate paired comparison isolates the marginal value of past-return features.
    rows += bootstrap(check[['text_trade_ge_0', 'with_returns_trade_ge_0']], 'text_trade_ge_0')
    pd.DataFrame(rows).to_csv(OUT/'bootstrap.csv', index=False)
    print('SELECTION', json.dumps(selected, indent=2), flush=True)
    print('MAIN BOOTSTRAP', pd.DataFrame(rows).query('block_days == 10').to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
