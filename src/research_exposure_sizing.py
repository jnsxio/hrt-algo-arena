"""Local exposure sizing on frozen Ridge + FinBERT validation forecasts.

Select multipliers on 2014/2015 walk-forward P&L; check selected settings on
the reused 2016 period. Never reads df_oos.pkl or writes a submission CSV.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import news
from research_baselines import metrics
from research_liquidity_bootstrap import bootstrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'exposure_sizing_results'
CSV = ROOT/'predictions_ridge_finbert.csv'
MODEL = 'finbert_probabilities_scale_0.1'
SPECS = {
    'unchanged': {'vol_power': 0., 'liquidity_power': 0.},
    'inverse_vol_sqrt': {'vol_power': .5, 'liquidity_power': 0.},
    'inverse_vol': {'vol_power': 1., 'liquidity_power': 0.},
    'inverse_variance': {'vol_power': 2., 'liquidity_power': 0.},
    'favor_lower_liquidity': {'vol_power': 0., 'liquidity_power': -.5},
    'favor_higher_liquidity': {'vol_power': 0., 'liquidity_power': .5},
    'inverse_vol_lower_liquidity': {'vol_power': 1., 'liquidity_power': -.5},
    'company_historical_risk': {'company_risk': True},
}


def multiplier(train, valid, spec):
    """Use only training estimates and current observable stock metadata."""
    if spec.get('company_risk'):
        # Shrink each ticker's historical squared target return toward the
        # training-wide level with 100 pseudo-observations. New tickers get 1.
        stats = train.returns.pow(2).groupby(level='symbol').agg(['sum', 'count'])
        global_m2 = float(train.returns.pow(2).mean())
        shrunk = (stats['sum'] + 100 * global_m2) / (stats['count'] + 100)
        risk = valid.index.get_level_values('symbol').map(shrunk).to_numpy(dtype=float)
        risk = np.where(np.isfinite(risk), risk, global_m2)
        result = np.sqrt(global_m2 / np.maximum(risk, 1e-8))
    else:
        vol_floor = max(float(train.vol.quantile(.05)), .001)
        liquidity_floor = max(float(train.weight.quantile(.05)), 1.)
        result = (float(train.vol.median()) / np.maximum(valid.vol.to_numpy(), vol_floor)) ** spec['vol_power']
        result *= (np.maximum(valid.weight.to_numpy(), liquidity_floor) / float(train.weight.median())) ** spec['liquidity_power']
    return np.clip(result, .25, 4.)


def main():
    OUT.mkdir(exist_ok=True)
    original_hash = hashlib.sha256(CSV.read_bytes()).hexdigest()
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None,
                            label='exposure sizing').get_data()
    years = df.index.get_level_values('date').year
    records, daily = [], {}
    chosen = None
    for year in [2014, 2015, 2016]:
        train, valid = df[years < year], df[years == year]
        predictions = pd.read_pickle(ROOT/'finbert_results'/f'validation_predictions_{year}.pkl')
        assert predictions.index.equals(valid.index)
        base = predictions[MODEL].to_numpy()
        candidates = list(SPECS) if year < 2016 else chosen['check_models']
        for name in candidates:
            m = multiplier(train, valid, SPECS[name])
            pred = base * m
            score, pnl = metrics(valid, pred)
            records.append({'year': year, 'model': name, **score,
                            'multiplier_min': float(m.min()), 'multiplier_median': float(np.median(m)),
                            'multiplier_max': float(m.max()),
                            'positive_multiplier': bool((m > 0).all())})
            daily[f'{year}_{name}'] = pnl
            assert np.array_equal(np.sign(base), np.sign(pred))
            print(f'{year} {name}: Sharpe={score["sharpe"]:.5f}', flush=True)
        if year == 2015:
            scores = pd.DataFrame(records).groupby('model').sharpe.mean().sort_values(ascending=False)
            best = scores.index[0]
            best_vol = scores[[n for n in scores.index if n in ['inverse_vol_sqrt', 'inverse_vol', 'inverse_variance']]].idxmax()
            best_liq = scores[[n for n in scores.index if n in ['favor_lower_liquidity', 'favor_higher_liquidity']]].idxmax()
            chosen = {'selected_model': best, 'spec': SPECS[best],
                      'selection_mean_sharpes': scores.to_dict(),
                      'selection_years': [2014, 2015], 'check_year': 2016,
                      'check_previously_seen': True,
                      'check_models': list(dict.fromkeys(['unchanged', best, best_vol, best_liq, 'company_historical_risk'])),
                      'all_candidate_specs': SPECS, 'multiplier_bounds': [.25, 4.],
                      'oos_accessed': False, 'submission_written': False,
                      'original_csv_sha256': original_hash}
            # Persist the decision before opening the 2016 prediction file.
            (OUT/'selection.json').write_text(json.dumps(chosen, indent=2))
        pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_pnl.csv')
    check = pd.DataFrame({name.removeprefix('2016_'): pnl for name, pnl in daily.items()
                          if name.startswith('2016_')})
    result = pd.DataFrame(bootstrap(check, 'unchanged'))
    result.to_csv(OUT/'bootstrap.csv', index=False)
    original_metrics = pd.read_csv(ROOT/'finbert_results/metrics.csv')
    for record in records:
        if record['model'] == 'unchanged':
            reference = original_metrics[(original_metrics.year == record['year']) &
                                         (original_metrics.model == MODEL)].sharpe.iloc[0]
            assert np.isclose(record['sharpe'], reference, rtol=1e-10)
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == original_hash
    (OUT/'verification.json').write_text(json.dumps({
        'baseline_scores_reproduced': True, 'submitted_candidate_csv_unchanged': True,
        'sha256': original_hash, 'directions_preserved': True}, indent=2))
    print(json.dumps(chosen, indent=2), flush=True)
    print(result.query('block_days == 10').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
