"""Local prediction clipping experiment; no OOS reads or submission creation.

Cap selected on mean 2014/2015 walk-forward Sharpe. Only the selected finite
cap and unchanged baseline are checked on the previously examined 2016 data.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import news
from research_baselines import metrics
from research_ridge_tuning import design, fit, mr_features, interactions
from research_liquidity_bootstrap import bootstrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'prediction_clipping_results'
CAPS = [None, .001, .0025, .005, .01, .02]


def predict(train, valid):
    a, b = design(train, valid, 'words13')
    zt, zv = mr_features(train, valid)
    _, pred = fit(interactions(a, zt, .25), interactions(b, zv, .25), train, 100.)
    return pred


def main():
    OUT.mkdir(exist_ok=True)
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None,
                            label='clipping research').get_data()
    years = df.index.get_level_values('date').year
    records, daily = [], {}
    selected = None
    for year in [2014, 2015, 2016]:
        train, valid = df[years < year], df[years == year]
        raw = predict(train, valid)
        caps = CAPS if year < 2016 else [None, selected['best_finite_cap']]
        for cap in caps:
            name = 'uncapped' if cap is None else f'cap_{cap:g}'
            pred = raw if cap is None else np.clip(raw, -cap, cap)
            score, pnl = metrics(valid, pred)
            records.append({'year': year, 'model': name, 'cap': cap,
                            'fraction_clipped': float(np.mean(pred != raw)), **score})
            daily[f'{year}_{name}'] = pnl
            print(f'{year} {name}: Sharpe={score["sharpe"]:.5f}, '
                  f'clipped={np.mean(pred != raw):.2%}', flush=True)
        # Cache local validation forecasts only, with labels for transparent diagnostics.
        diagnostic = valid[['headline', 'returns', 'weight']].copy()
        diagnostic['uncapped_prediction'] = raw
        diagnostic.to_pickle(OUT/f'validation_{year}.pkl')
        if year == 2015:
            table = pd.DataFrame(records)
            means = table.groupby('model').sharpe.mean().sort_values(ascending=False)
            finite = means.drop('uncapped').index[0]
            # Prefer the unchanged baseline when scores tie.
            best = 'uncapped' if np.isclose(means['uncapped'], means.iloc[0], rtol=0, atol=1e-12) else means.index[0]
            selected = {'caps': CAPS, 'best_overall': best,
                        'best_finite_cap': float(finite.split('_')[1]),
                        'selection_mean_sharpes': means.to_dict(),
                        'selection_years': [2014, 2015], 'check_year': 2016,
                        'check_previously_seen': True, 'oos_accessed': False,
                        'submission_written': False,
                        'model': 'words13, Ridge alpha=100, past-return interaction strength=.25'}
            (OUT/'selection.json').write_text(json.dumps(selected, indent=2))
    table = pd.DataFrame(records)
    table.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_pnl.csv')
    check = pd.DataFrame({name.removeprefix('2016_'): pnl for name, pnl in daily.items()
                          if name.startswith('2016_')})
    result = pd.DataFrame(bootstrap(check, 'uncapped'))
    result.to_csv(OUT/'bootstrap.csv', index=False)
    # Verify uncapped values reproduce the previously saved experiment.
    previous = pd.read_csv(ROOT/'liquidity_bootstrap_results'/'metrics.csv')
    for year in [2014, 2015, 2016]:
        reference = previous[(previous.year == year) & (previous.model == 'with_returns_trade_ge_0')].sharpe.iloc[0]
        actual = table[(table.year == year) & (table.model == 'uncapped')].sharpe.iloc[0]
        assert np.isclose(actual, reference, rtol=1e-9), (year, actual, reference)
    print('Uncapped baseline matches previous results for all years.', flush=True)
    print(json.dumps(selected, indent=2), flush=True)
    print(result[result.model != 'uncapped'].to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
