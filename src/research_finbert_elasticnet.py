"""Local FinBERT feature and ElasticNet experiments. No OOS reads/submissions.

python research_finbert_elasticnet.py elasticnet
python research_finbert_elasticnet.py finbert  # after the sentiment cache completes
"""
import argparse
import json
from pathlib import Path
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import news
from research_baselines import metrics
from research_ridge_tuning import design, mr_features, interactions
from research_liquidity_bootstrap import bootstrap

ROOT = Path(__file__).resolve().parent


def fit_ridge(a, b, y, alpha=100):
    model = Ridge(alpha=alpha, solver='lsqr', tol=1e-6).fit(a, y)
    return model.predict(b)


def fit_en(a, b, y, l2, l1):
    # Ridge objective: sum(error^2) + l2*sum(coef^2).
    # EN objective: mean(error^2)/2 + l1*sum(abs(coef)) + (l2/n)*sum(coef^2)/2.
    alpha = l1 + l2 / len(y)
    model = ElasticNet(alpha=alpha, l1_ratio=l1/alpha, max_iter=3000, tol=1e-6,
                       selection='cyclic')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(a.tocsc(), y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not converged:
        raise RuntimeError(f'ElasticNet failed to converge: l2={l2}, l1={l1}')
    return model.predict(b), {'nonzero_coefficients': int(np.count_nonzero(model.coef_)),
                             'total_coefficients': len(model.coef_), 'iterations': int(model.n_iter_),
                             'dual_gap': float(model.dual_gap_), 'alpha': alpha,
                             'l1_ratio': l1/alpha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['elasticnet', 'finbert'])
    args = parser.parse_args()
    out = ROOT / (args.mode + '_results')
    out.mkdir(exist_ok=True)
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None,
                            label=args.mode).get_data()
    years = df.index.get_level_values('date').year
    sentiment = None
    if args.mode == 'finbert':
        with sqlite3.connect(ROOT/'finbert_results'/'sentiment_cache.sqlite') as db:
            sentiment = pd.read_sql_query('SELECT * FROM sentiment', db).set_index('headline')
        assert set(df.headline).issubset(sentiment.index), 'FinBERT cache incomplete'
        assert np.allclose(sentiment.sum(axis=1), 1, atol=1e-5)
    specs, records, daily = {}, [], {}
    selected = None
    for year in [2014, 2015, 2016]:
        train, valid = df[years < year], df[years == year]
        xt, xv = design(train, valid, 'words13')
        zt, zv = mr_features(train, valid)
        a, b = interactions(xt, zt, .25), interactions(xv, zv, .25)
        base = fit_ridge(a, b, train.returns)
        forecasts = {'baseline': base}
        def record(name, pred, info=None):
            scores, pnl = metrics(valid, pred)
            records.append({'year': year, 'model': name, **scores, **(info or {})})
            daily[f'{year}_{name}'] = pnl
            forecasts[name] = pred
            print(f'{args.mode} {year} {name}: Sharpe={scores["sharpe"]:.5f} {info or ""}', flush=True)
        record('baseline', base)
        if args.mode == 'elasticnet':
            candidates = [(l2, l1) for l2 in [30., 100.] for l1 in [1e-6, 1e-5, 3e-5]]
            if selected is not None:
                candidates = [tuple(selected['spec'])]
            for l2, l1 in candidates:
                name = f'en_l2_{l2:g}_l1_{l1:g}'
                specs[name] = [l2, l1]
                pred, info = fit_en(a, b, train.returns, l2, l1)
                record(name, pred, info)
        else:
            st = sentiment.loc[train.headline, ['positive', 'negative', 'neutral']].to_numpy()
            sv = sentiment.loc[valid.headline, ['positive', 'negative', 'neutral']].to_numpy()
            record('raw_sentiment', sv[:, 0] - sv[:, 1])
            # All transformations and return calibration fit on earlier years only.
            scaler = StandardScaler().fit(st)
            st, sv = scaler.transform(st), scaler.transform(sv)
            compact_t = np.column_stack([st, zt, st * zt[:, 0:1], st * zt[:, 1:2]])
            compact_v = np.column_stack([sv, zv, sv * zv[:, 0:1], sv * zv[:, 1:2]])
            record('finbert_calibrated', fit_ridge(compact_t, compact_v, train.returns))
            candidates = [(kind, scale) for kind in ['probabilities', 'probabilities_and_returns'] for scale in [.1, 1.]]
            if selected is not None:
                candidates = [tuple(selected['spec'])]
            for kind, scale in candidates:
                if kind == 'probabilities':
                    ft, fv = st, sv
                else:
                    ft = np.column_stack([st, st * zt[:, 0:1], st * zt[:, 1:2]])
                    fv = np.column_stack([sv, sv * zv[:, 0:1], sv * zv[:, 1:2]])
                name = f'finbert_{kind}_scale_{scale:g}'
                specs[name] = [kind, scale]
                combined_t = hstack([a, csr_matrix(ft * scale)], format='csr')
                combined_v = hstack([b, csr_matrix(fv * scale)], format='csr')
                record(name, fit_ridge(combined_t, combined_v, train.returns))
        if year == 2015:
            table = pd.DataFrame(records)
            means = table[table.model.isin(specs)].groupby('model').sharpe.mean().sort_values(ascending=False)
            best = means.index[0]
            selected = {'mode': args.mode, 'selected_model': best, 'spec': specs[best],
                        'selection_mean_sharpes': means.to_dict(), 'selection_years': [2014, 2015],
                        'check_year': 2016, 'check_previously_seen': True,
                        'oos_accessed': False, 'submission_written': False}
            (out/'selection.json').write_text(json.dumps(selected, indent=2))
        # Store validation predictions for future paired tests; these are not submissions.
        pd.DataFrame(forecasts, index=valid.index).to_pickle(out/f'validation_predictions_{year}.pkl')
        pd.DataFrame(records).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(out/'daily_pnl.csv')
    check = pd.DataFrame({name.removeprefix('2016_'): pnl for name, pnl in daily.items()
                          if name.startswith('2016_')})
    result = pd.DataFrame(bootstrap(check, 'baseline'))
    result.to_csv(out/'bootstrap.csv', index=False)
    old = pd.read_csv(ROOT/'liquidity_bootstrap_results'/'metrics.csv')
    for year in [2014, 2015, 2016]:
        reference = old[(old.year == year) & (old.model == 'with_returns_trade_ge_0')].sharpe.iloc[0]
        actual = next(r['sharpe'] for r in records if r['year'] == year and r['model'] == 'baseline')
        # LSQR can differ in the last decimals with/without explicit unit sample
        # weights; this tolerance is far below reported score precision.
        assert np.isclose(actual, reference, rtol=0, atol=1e-6)
    print('Baseline matches prior experiment.', flush=True)
    print(json.dumps(selected, indent=2), flush=True)
    print(result.query('block_days == 10').to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
