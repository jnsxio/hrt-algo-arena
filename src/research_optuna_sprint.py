"""Time-limited local Optuna search. Generates separate candidates; no uploads.

Tuning uses only 2014/2015 OOF predictions. Reused 2016 is reported afterward.
Headline-density thinning uses fixed masks so trials are directly comparable.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import optuna
import news
from research_baselines import metrics
from research_liquidity_bootstrap import bootstrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'optuna_sprint_results'
MODEL = 'finbert_probabilities_scale_0.1'
INCUMBENT = dict(vol_power=1., liquidity_power=-1., lower=.25, upper=4., offset=0.)


def prepare(train, valid, raw, density, seed=44):
    vol = np.maximum(valid.vol.to_numpy(), max(float(train.vol.quantile(.05)), .001))
    weight = np.maximum(valid.weight.to_numpy(), max(float(train.weight.quantile(.05)), 1.))
    codes, dates = pd.factorize(valid.index.get_level_values('date'), sort=True)
    f = min(1., density/(len(valid)/len(dates)))
    rng = np.random.default_rng(seed)
    return {'raw': raw, 'lv': np.log(float(train.vol.median())/vol),
            'lw': np.log(weight/float(train.weight.median())),
            'pnl_factor': valid.returns.to_numpy()*valid.weight.to_numpy() if 'returns' in valid else None,
            'codes': codes, 'days': len(dates),
            'masks': [rng.random(len(valid)) < f for _ in range(12)]}


def predictions(data, p):
    mult = np.clip(np.exp(p['vol_power']*data['lv'] + p['liquidity_power']*data['lw']), p['lower'], p['upper'])
    return (data['raw'] + p.get('offset', 0.)) * mult


def sharpe(pnl):
    return float(np.sqrt(252)*pnl.mean()/(pnl.std(ddof=1)+1e-12))


def evaluate(data, p):
    row_pnl = predictions(data, p)*data['pnl_factor']
    pnl = np.bincount(data['codes'], weights=row_pnl, minlength=data['days'])
    sparse = [sharpe(np.bincount(data['codes'], weights=row_pnl*mask, minlength=data['days'])) for mask in data['masks']]
    return sharpe(pnl), float(np.mean(sparse))


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    protected = sorted(ROOT.glob('predictions_ridge_finbert*.csv'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    df, test = news.get_ins_data(), news.get_oos_data()
    assert 'returns' not in test
    density = float(test.groupby(level='date').size().mean())
    years = df.index.get_level_values('date').year
    tuning = {}
    for year in [2014, 2015]:
        train, valid = df[years < year], df[years == year]
        stored = pd.read_pickle(ROOT/'finbert_results'/f'validation_predictions_{year}.pkl')
        assert stored.index.equals(valid.index)
        tuning[year] = prepare(train, valid, stored[MODEL].to_numpy(), density)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selections = {'incumbent': INCUMBENT}
    summaries = {}
    for mode in ['sizing', 'sizing_offset']:
        def objective(trial):
            p = {'vol_power': trial.suggest_float('vol_power', .25, 2.),
                 'liquidity_power': trial.suggest_float('liquidity_power', -1.75, -.25),
                 'lower': trial.suggest_float('lower', .05, .4, log=True),
                 'upper': trial.suggest_float('upper', 2., 10., log=True),
                 'offset': trial.suggest_float('offset', -.0015, .0015) if mode == 'sizing_offset' else 0.}
            scores = np.array([evaluate(d, p) for d in tuning.values()])
            # Aim at sparse daily portfolios and penalize instability across years.
            value = scores[:, 1].mean() - .15*scores[:, 1].std()
            trial.set_user_attr('full_sharpes', scores[:, 0].tolist())
            trial.set_user_attr('density_sharpes', scores[:, 1].tolist())
            return float(value)
        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=20),
                                   study_name=f'algoarena_{mode}')
        seed_params = {k: v for k, v in INCUMBENT.items() if mode == 'sizing_offset' or k != 'offset'}
        study.enqueue_trial(seed_params)
        if mode == 'sizing_offset':
            study.enqueue_trial(selections['optuna_sizing'])
        study.optimize(objective, n_trials=120, timeout=75.)
        best = dict(study.best_params)
        best.setdefault('offset', 0.)
        selections[f'optuna_{mode}'] = best
        summaries[mode] = {'parameters': best, 'objective': study.best_value,
                           'trials': len(study.trials), 'validation': study.best_trial.user_attrs}
        study.trials_dataframe().to_csv(OUT/f'{mode}_trials.csv', index=False)
        print(mode, json.dumps(summaries[mode]), flush=True)
    # A controlled offset-only probe separates forecast calibration from risk sizing.
    selections['offset_only_probe'] = dict(INCUMBENT, offset=selections['optuna_sizing_offset']['offset'])
    summary = {'selection_years': [2014, 2015], 'check_year': 2016,
               'check_previously_seen': True, 'density': density, 'studies': summaries,
               'parameters': selections, 'uploaded': False,
               'leaderboard_reference_reported_by_user': 4.838}
    (OUT/'selection.json').write_text(json.dumps(summary, indent=2))
    records, checks = [], {}
    for name, params in selections.items():
        for year, data in tuning.items():
            full, sparse = evaluate(data, params)
            records.append({'year': year, 'model': name, 'sharpe': full, 'density_sharpe': sparse})
    valid = df[years == 2016]
    stored = pd.read_pickle(ROOT/'finbert_results/validation_predictions_2016.pkl')
    assert stored.index.equals(valid.index)
    check_data = prepare(df[years < 2016], valid, stored[MODEL].to_numpy(), density, seed=2027)
    for name, params in selections.items():
        p = predictions(check_data, params)
        full, sparse = evaluate(check_data, params)
        records.append({'year': 2016, 'model': name, 'sharpe': full, 'density_sharpe': sparse})
        checks[name] = metrics(valid, p)[1]
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/'daily_pnl_2016.csv')
    pd.DataFrame(bootstrap(pd.DataFrame(checks), 'incumbent', repetitions=2000)).to_csv(OUT/'bootstrap.csv', index=False)
    source = pd.read_csv(ROOT/'predictions_ridge_finbert.csv', dtype={'headline_id': 'int64', 'pred': 'float64'}, float_precision='round_trip')
    assert np.array_equal(source.headline_id.to_numpy(), test.headline_id.to_numpy())
    target = prepare(df, test, source.pred.to_numpy(), density)
    manifest = {}
    for name, params in selections.items():
        if name == 'incumbent':
            continue
        destination = ROOT/f'predictions_{name}.csv'
        p = predictions(target, params)
        assert np.isfinite(p).all()
        news.write_submission(prediction_df=test.assign(pred=p), path=destination)
        written = pd.read_csv(destination, dtype={'headline_id': 'int64', 'pred': 'float64'}, float_precision='round_trip')
        assert list(written.columns) == ['headline_id', 'pred']
        assert np.array_equal(written.headline_id.to_numpy(), test.headline_id.to_numpy())
        assert written.headline_id.is_unique and np.array_equal(written.pred.to_numpy(), p)
        manifest[name] = {'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                          'parameters': params, 'rows': len(written),
                          'changed_sign_fraction': float(np.mean(np.sign(p) != np.sign(source.pred))),
                          'uploaded': False}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print('MANIFEST', json.dumps(manifest, indent=2), flush=True)
    print('Elapsed seconds', round(time.monotonic()-start, 1), flush=True)


if __name__ == '__main__':
    main()
