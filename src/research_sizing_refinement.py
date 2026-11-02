"""Refine sizing and then Ridge using the FINAL daily-P&L objective.

All choices use 2014/2015 chronological validation. 2016 is a reused check.
OOS metadata are used only to measure headline density, never to fit or tune.
Existing submission CSVs are read only for hashes and never changed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import news
from research_baselines import metrics
from research_exposure_sizing import multiplier
from research_ridge_tuning import design, mr_features, interactions
from research_liquidity_bootstrap import bootstrap
from make_ridge_finbert_csv import get_sentiment

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'sizing_refinement_results'
OLD = {'vol_power': 1., 'liquidity_power': -.5}
FB_MODEL = 'finbert_probabilities_scale_0.1'
GRID = {f'vol{v:g}_liq{l:g}': {'vol_power': v, 'liquidity_power': l}
        for v in [.5, 1., 1.5, 2.] for l in [-.25, -.5, -.75, -1.]}


def matrices(train, valid, sentiment):
    xt, xv = design(train, valid, 'words13')
    zt, zv = mr_features(train, valid)
    a, b = interactions(xt, zt, .25), interactions(xv, zv, .25)
    columns = ['positive', 'negative', 'neutral']
    st, sv = sentiment.loc[train.headline, columns].to_numpy(), sentiment.loc[valid.headline, columns].to_numpy()
    scaler = StandardScaler().fit(st)
    return a, b, scaler.transform(st), scaler.transform(sv)


def predict(mats, y, alpha, fb_scale):
    a, b, st, sv = mats
    if fb_scale:
        a = hstack([a, csr_matrix(st*fb_scale)], format='csr')
        b = hstack([b, csr_matrix(sv*fb_scale)], format='csr')
    return Ridge(alpha=alpha, solver='lsqr', tol=1e-6).fit(a, y).predict(b)


def main():
    OUT.mkdir(exist_ok=True)
    protected = [ROOT/'predictions_ridge_finbert.csv', ROOT/'predictions_ridge_finbert_sized.csv']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None, label='research').get_data()
    test_metadata = news.PredictionTask(dataset_source=ROOT/'df_oos.pkl', error_manager=None, label='density only').get_data()
    assert 'returns' not in test_metadata.columns
    years = df.index.get_level_values('date').year
    density = {'train': {}, 'test': {}}
    for split, data in [('train', df), ('test', test_metadata)]:
        for year, subset in data.groupby(data.index.get_level_values('date').year):
            density[split][str(year)] = {'rows': len(subset), 'days': subset.index.get_level_values('date').nunique(),
                                       'headlines_per_day': float(subset.groupby(level='date').size().mean())}
    target_density = float(test_metadata.groupby(level='date').size().mean())
    (OUT/'headline_density.json').write_text(json.dumps(density, indent=2))
    records, daily = [], {}
    def record(stage, year, name, valid, pred):
        score, pnl = metrics(valid, pred)
        records.append({'stage': stage, 'year': year, 'model': name, **score})
        daily[f'{year}_{stage}_{name}'] = pnl
        print(f'{stage} {year} {name}: {score["sharpe"]:.5f}', flush=True)
    # First select risk exponents, keeping the existing return forecasts frozen.
    for year in [2014, 2015]:
        train, valid = df[years < year], df[years == year]
        cached = pd.read_pickle(ROOT/'finbert_results'/f'validation_predictions_{year}.pkl')
        assert cached.index.equals(valid.index)
        raw = cached[FB_MODEL].to_numpy()
        for name, spec in GRID.items():
            record('sizing', year, name, valid, raw * multiplier(train, valid, spec))
    table = pd.DataFrame(records)
    sizing_scores = table.groupby('model').sharpe.mean().sort_values(ascending=False)
    best_sizing = sizing_scores.index[0]
    spec = GRID[best_sizing]
    selection = {'sizing_model': best_sizing, 'sizing_spec': spec,
                 'sizing_selection_scores': sizing_scores.to_dict(),
                 'selection_years': [2014, 2015], 'check_year': 2016, 'check_previously_seen': True,
                 'oos_metadata_used_for_density_only': True, 'submission_written': False}
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2))
    # Retune a small model grid against the selected position-sizing objective.
    sentiment = get_sentiment(ROOT/'finbert_results/sentiment_cache.sqlite')
    model_specs = {}
    for year in [2014, 2015]:
        train, valid = df[years < year], df[years == year]
        mats = matrices(train, valid, sentiment)
        scale = multiplier(train, valid, spec)
        for alpha in [30., 100., 300.]:
            for fb_scale in [0., .1]:
                name = f'alpha{alpha:g}_fb{fb_scale:g}'
                model_specs[name] = {'alpha': alpha, 'fb_scale': fb_scale}
                pred = predict(mats, train.returns, alpha, fb_scale)
                record('model', year, name, valid, pred * scale)
    table = pd.DataFrame(records)
    model_scores = table[table.stage == 'model'].groupby('model').sharpe.mean().sort_values(ascending=False)
    best_model = model_scores.index[0]
    selection.update({'return_model': best_model, 'return_model_spec': model_specs[best_model],
                      'model_selection_scores': model_scores.to_dict()})
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2))
    # Only now evaluate the selected candidates on the reused check year.
    year = 2016
    train, valid = df[years < year], df[years == year]
    cached = pd.read_pickle(ROOT/'finbert_results/validation_predictions_2016.pkl')
    assert cached.index.equals(valid.index)
    old_raw = cached[FB_MODEL].to_numpy()
    candidates = {'current_sized': old_raw * multiplier(train, valid, OLD),
                  'refined_sizing': old_raw * multiplier(train, valid, spec)}
    mats = matrices(train, valid, sentiment)
    new_raw = predict(mats, train.returns, **model_specs[best_model])
    candidates['refined_model_and_sizing'] = new_raw * multiplier(train, valid, spec)
    for name, pred in candidates.items():
        record('check', year, name, valid, pred)
    check = pd.DataFrame({name: daily[f'2016_check_{name}'] for name in candidates})
    pd.DataFrame(bootstrap(check, 'current_sized')).to_csv(OUT/'bootstrap.csv', index=False)
    # Paired Bernoulli thinning is a headline-density stress test, not an estimate
    # of the hidden score or a confidence interval from independent new data.
    dates = valid.index.get_level_values('date')
    codes, unique_dates = pd.factorize(dates, sort=True)
    n_days = len(unique_dates)
    fraction = min(1., target_density / (len(valid)/n_days))
    pnl_rows = {name: pred * valid.returns.to_numpy() * valid.weight.to_numpy()
                for name, pred in candidates.items()}
    rng = np.random.default_rng(2026)
    thinning = []
    for repeat in range(200):
        keep = rng.random(len(valid)) < fraction
        for name, pnl in pnl_rows.items():
            # Retain all original dates, including any zero-event days.
            p = np.bincount(codes, weights=pnl*keep, minlength=n_days)
            score = np.sqrt(252)*p.mean()/(p.std(ddof=1)+1e-12)
            thinning.append({'repeat': repeat, 'model': name, 'sharpe': score,
                             'headlines_per_day': float(keep.sum()/n_days)})
    pd.DataFrame(thinning).to_csv(OUT/'density_stress.csv', index=False)
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_pnl.csv')
    pd.DataFrame(candidates, index=valid.index).to_pickle(OUT/'validation_predictions_2016.pkl')
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    print(json.dumps(selection, indent=2), flush=True)
    print('Density stress means:', pd.DataFrame(thinning).groupby('model').sharpe.mean().to_dict(), flush=True)
    print(pd.read_csv(OUT/'bootstrap.csv').query('block_days == 10').to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
