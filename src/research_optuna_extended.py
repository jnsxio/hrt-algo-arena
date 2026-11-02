"""Extended local sizing search with fresh-resample verification; no uploads.

1200 TPE trials, 48 paired density masks per tuning year, top 12 verified on
256 fresh masks per year. 2016 is evaluated only after freezing the selection.
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
from research_optuna_sprint import prepare, predictions, evaluate, INCUMBENT, MODEL

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'optuna_extended_results'


def resample_masks(data, count, target_density, seed):
    rng = np.random.default_rng(seed)
    fraction = min(1., target_density/(len(data['raw'])/data['days']))
    data['masks'] = [rng.random(len(data['raw'])) < fraction for _ in range(count)]


def objective_value(scores):
    sparse = np.array(scores)[:, 1]
    return float(sparse.mean() - .15*sparse.std())


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    protected = sorted(ROOT.glob('predictions_*.csv'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    df, test = news.get_ins_data(), news.get_oos_data()
    density = float(test.groupby(level='date').size().mean())
    years = df.index.get_level_values('date').year
    tuning = {}
    for year in [2014, 2015]:
        train, valid = df[years < year], df[years == year]
        cached = pd.read_pickle(ROOT/'finbert_results'/f'validation_predictions_{year}.pkl')
        assert cached.index.equals(valid.index)
        tuning[year] = prepare(train, valid, cached[MODEL].to_numpy(), density)
        resample_masks(tuning[year], 48, density, seed=year+77)
    old = json.loads((ROOT/'optuna_sprint_results/selection.json').read_text())['parameters']['optuna_sizing']
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=77, n_startup_trials=60))
    for p in [INCUMBENT, old]:
        study.enqueue_trial({k:v for k,v in p.items() if k != 'offset'})
    def objective(trial):
        p = {'vol_power': trial.suggest_float('vol_power', .25, 2.),
             'liquidity_power': trial.suggest_float('liquidity_power', -1.75, -.25),
             'lower': trial.suggest_float('lower', .05, .4, log=True),
             'upper': trial.suggest_float('upper', 2., 10., log=True), 'offset': 0.}
        scores = [evaluate(d, p) for d in tuning.values()]
        trial.set_user_attr('full_sharpes', [x[0] for x in scores])
        trial.set_user_attr('density_sharpes', [x[1] for x in scores])
        if trial.number % 100 == 0:
            print(f'Trial {trial.number}; elapsed {time.monotonic()-start:.1f}s', flush=True)
        return objective_value(scores)
    study.optimize(objective, n_trials=1200, timeout=180.)
    study.trials_dataframe().to_csv(OUT/'trials.csv', index=False)
    top = sorted([t for t in study.trials if t.value is not None], key=lambda t:t.value, reverse=True)[:12]
    candidate_params = {'incumbent_v2': INCUMBENT, 'previous_optuna': old}
    for t in top:
        candidate_params[f'trial_{t.number}'] = dict(t.params, offset=0.)
    # New masks guard against picking a winner due only to search-mask noise.
    for year, data in tuning.items():
        resample_masks(data, 256, density, seed=year+9900)
    checked = []
    for name, p in candidate_params.items():
        scores = [evaluate(d, p) for d in tuning.values()]
        checked.append({'name': name, 'objective': objective_value(scores),
                        'sharpe_2014': scores[0][0], 'sharpe_2015': scores[1][0],
                        'density_2014': scores[0][1], 'density_2015': scores[1][1]})
    verification = pd.DataFrame(checked).sort_values('objective', ascending=False)
    verification.to_csv(OUT/'fresh_resample_verification.csv', index=False)
    chosen_name = verification.iloc[0]['name']
    best = candidate_params[chosen_name]
    # A midpoint is a bounded, less aggressive alternative for sequential uploads.
    midpoint = {k: (INCUMBENT[k]+best[k])/2 for k in best}
    choices = {'incumbent_v2': INCUMBENT, 'previous_optuna': old,
               'optuna_extended': best, 'optuna_midpoint': midpoint}
    selection = {'selected_trial': chosen_name, 'parameters': choices,
                 'trials_completed': len(study.trials), 'search_resamples_per_year': 48,
                 'fresh_resamples_per_year': 256, 'tuning_years': [2014, 2015],
                 'check_year': 2016, 'check_previously_seen': True, 'uploaded': False}
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2))
    print('Selection frozen:', json.dumps(selection), flush=True)
    records, daily = [], {}
    for year in [2014, 2015, 2016]:
        train, valid = df[years < year], df[years == year]
        if year == 2016:
            cached = pd.read_pickle(ROOT/'finbert_results/validation_predictions_2016.pkl')
            assert cached.index.equals(valid.index)
            data = prepare(train, valid, cached[MODEL].to_numpy(), density)
            resample_masks(data, 512, density, seed=199)
        else:
            data = tuning[year]
        for name, p in choices.items():
            full, sparse = evaluate(data, p)
            records.append({'year': year, 'model': name, 'sharpe': full, 'density_sharpe': sparse})
            if year == 2016:
                daily[name] = metrics(valid, predictions(data,p))[1]
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_pnl_2016.csv')
    pd.DataFrame(bootstrap(pd.DataFrame(daily), 'incumbent_v2', repetitions=10000)).to_csv(OUT/'bootstrap.csv', index=False)
    source = pd.read_csv(ROOT/'predictions_ridge_finbert.csv', dtype={'headline_id':'int64', 'pred':'float64'},float_precision='round_trip')
    assert np.array_equal(source.headline_id.to_numpy(), test.headline_id.to_numpy())
    target = prepare(df, test, source.pred.to_numpy(), density)
    manifest = {}
    for name in ['optuna_extended', 'optuna_midpoint']:
        p = predictions(target, choices[name])
        assert np.isfinite(p).all()
        destination = ROOT/f'predictions_{name}.csv'
        news.write_submission(prediction_df=test.assign(pred=p), path=destination)
        result = pd.read_csv(destination, dtype={'headline_id':'int64', 'pred':'float64'},float_precision='round_trip')
        assert result.headline_id.is_unique and np.array_equal(result.headline_id.to_numpy(), test.headline_id.to_numpy())
        assert list(result.columns) == ['headline_id','pred'] and np.array_equal(result.pred.to_numpy(), p)
        manifest[name] = {'path':str(destination), 'rows':len(result),
                          'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(), 'uploaded':False}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in protected:
        assert hashes[str(p)] == hashlib.sha256(p.read_bytes()).hexdigest()
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print(pd.read_csv(OUT/'bootstrap.csv').query('block_days==10').to_string(index=False), flush=True)
    print('Elapsed seconds:',time.monotonic()-start,flush=True)


if __name__ == '__main__':
    main()
