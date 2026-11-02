"""Local-only Ridge tuning. Reads df_ins.pkl only; never writes a submission.

Select on expanding-window 2014 and 2015 validation; report previously seen
2016 as a follow-up check, not a pristine holdout. Run with the research venv.
"""
from pathlib import Path
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'algoarena-matplotlib'))
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import news
from research_baselines import metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ridge_tuning_results'
ALPHAS = [30., 100., 300., 1000.]
CONFIGS = {'words12': (1, 2), 'words13': (1, 3), 'words1': (1, 1)}


def design(train, valid, config):
    v = TfidfVectorizer(ngram_range=CONFIGS[config], min_df=5,
                       max_features=100000, sublinear_tf=True)
    return v.fit_transform(train.headline), v.transform(valid.headline)


def fit(a, b, train, alpha, weighted=False):
    m = Ridge(alpha=alpha, solver='lsqr', tol=1e-6)
    weights = train.weight.to_numpy() if weighted else np.ones(len(train))
    m.fit(a, train.returns, sample_weight=weights / weights.mean())
    return m.predict(a), m.predict(b)


def mr_features(train, valid):
    cols = ['returns_past_hour', 'returns_past_day']
    floor = max(float(train.vol.quantile(.05)), .001)
    def transform(d):
        return np.clip(d[cols].to_numpy() / np.maximum(d.vol.to_numpy()[:, None], floor), -3, 3)
    return transform(train), transform(valid)


def interactions(x, z, strength):
    return hstack([x, csr_matrix(z), x.multiply(z[:, 0:1]) * strength,
                   x.multiply(z[:, 1:2]) * strength], format='csr')


def main():
    OUT.mkdir(exist_ok=True)
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None,
                            label='research').get_data()
    years = df.index.get_level_values('date').year
    splits = {y: (df[years < y], df[years == y]) for y in [2014, 2015, 2016]}
    records, daily, predictions, cache = [], {}, {}, {}
    def record(stage, year, name, valid, p):
        scores, pnl = metrics(valid, p)
        records.append(dict(stage=stage, year=year, model=name, **scores))
        daily[f'{year}_{name}'] = pnl
        predictions[(year, name)] = p
        print(f'{stage} {year} {name}: {scores["sharpe"]:.4f}', flush=True)
    def choose(stage):
        scores = pd.DataFrame(records)
        scores = scores[(scores.stage == stage) & (scores.year < 2016)]
        return scores.groupby('model').sharpe.mean().idxmax()
    # Stage 1: a small, fixed text grid. No 2016 scores used for selection.
    specs = {}
    for year in [2014, 2015]:
        train, valid = splits[year]
        for config in CONFIGS:
            a, b = design(train, valid, config)
            cache[(year, config)] = (a, b)
            for alpha in ALPHAS:
                name = f'{config}_a{alpha:g}'
                specs[name] = (config, alpha)
                _, p = fit(a, b, train, alpha)
                record('text_grid', year, name, valid, p)
    best_text = choose('text_grid')
    config, alpha = specs[best_text]
    # Stage 2: prespecified liquidity-weighted fit, MR blend and word/MR interactions.
    # Blend normalization uses training features/predictions only, never validation labels.
    for year in [2014, 2015]:
        train, valid = splits[year]
        a, b = cache[(year, config)]
        pt, p = fit(a, b, train, alpha)
        zt, zv = mr_features(train, valid)
        record('extensions', year, 'text', valid, p)
        _, wp = fit(a, b, train, alpha, weighted=True)
        record('extensions', year, 'liquidity_weighted_text', valid, wp)
        scale = np.sqrt(np.mean(pt ** 2))
        for j, horizon in enumerate(['hour', 'day']):
            reversion = -zv[:, j] / max(np.sqrt(np.mean(zt[:, j] ** 2)), 1e-12) * scale
            for w in [.1, .25, .5]:
                record('extensions', year, f'blend_{horizon}_{w:g}', valid, p + w * reversion)
        for strength in [.25, 1.]:
            _, ip = fit(interactions(a, zt, strength), interactions(b, zv, strength), train, alpha)
            record('extensions', year, f'interactions_{strength:g}', valid, ip)
    best_extension = choose('extensions')
    summary = {'selection_years': [2014, 2015], 'selection_metric': 'mean yearly daily-PnL Sharpe',
               'check_year': 2016, 'check_year_previously_seen': True,
               'best_text': best_text, 'best_extension': best_extension,
               'oos_accessed': False, 'submission_written': False,
               'alphas': ALPHAS, 'text_configs': CONFIGS}
    # Freeze and save selections before the follow-up check.
    (OUT/'selection.json').write_text(json.dumps(summary, indent=2))
    year = 2016
    train, valid = splits[year]
    a, b = design(train, valid, 'words12')
    _, old = fit(a, b, train, 100.)
    record('check', year, 'original_text_a100', valid, old)
    if config != 'words12':
        a, b = design(train, valid, config)
    pt, p = fit(a, b, train, alpha)
    record('check', year, 'text', valid, p)
    zt, zv = mr_features(train, valid)
    # Evaluate prespecified representative MR mechanisms, regardless of the winner.
    _, wp = fit(a, b, train, alpha, weighted=True)
    record('check', year, 'liquidity_weighted_text', valid, wp)
    scale = np.sqrt(np.mean(pt ** 2))
    extension_scores = pd.DataFrame(records)
    extension_scores = extension_scores[extension_scores.stage == 'extensions'].groupby('model').sharpe.mean()
    best_blend = extension_scores[extension_scores.index.str.startswith('blend_')].idxmax()
    _, horizon, w = best_blend.split('_')
    j = ['hour', 'day'].index(horizon)
    mr = -zv[:, j] / max(np.sqrt(np.mean(zt[:, j] ** 2)), 1e-12) * scale
    record('check', year, best_blend, valid, p + float(w) * mr)
    best_interaction = extension_scores[extension_scores.index.str.startswith('interactions_')].idxmax()
    strength = float(best_interaction.split('_')[1])
    _, ip = fit(interactions(a, zt, strength), interactions(b, zv, strength), train, alpha)
    record('check', year, best_interaction, valid, ip)
    # Duplicate-text diagnostic on the frozen text model; not used to select it.
    unseen = ~valid.headline.isin(set(train.headline))
    summary['2016_text_unseen_headlines'] = metrics(valid[unseen], p[unseen])[0]
    # Paired resampling of calendar-week P&Ls: descriptive uncertainty, not a new tuning score.
    baseline = daily['2016_original_text_a100']
    improved = daily['2016_text']
    weekly_groups = baseline.index.to_period('W')
    blocks = [np.flatnonzero(weekly_groups == w) for w in weekly_groups.unique()]
    rng = np.random.default_rng(42)
    comparisons = {'text_vs_original': (daily['2016_text'], baseline),
                   'interactions_vs_original': (daily[f'2016_{best_interaction}'], baseline),
                   'interactions_vs_tuned_text': (daily[f'2016_{best_interaction}'], improved)}
    differences = {name: [] for name in comparisons}
    def sharpe(x):
        return np.sqrt(252) * np.mean(x) / (np.std(x, ddof=1) + 1e-12)
    for _ in range(2000):
        ix = np.concatenate([blocks[k] for k in rng.integers(len(blocks), size=len(blocks))])
        for name, (candidate, reference) in comparisons.items():
            differences[name].append(sharpe(candidate.to_numpy()[ix]) - sharpe(reference.to_numpy()[ix]))
    summary['2016_sharpe_difference_week_bootstrap_95pct'] = {
        name: np.quantile(values, [.025, .975]).tolist() for name, values in differences.items()}
    summary['best_mr_blend'] = best_blend
    summary['best_interaction'] = best_interaction
    (OUT/'selection.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(records).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_pnl.csv')
    # Local diagnostic forecasts only, explicitly named and including validation labels.
    diagnostic = valid[['headline', 'returns', 'weight']].copy()
    diagnostic['original_text'] = old
    diagnostic['optimized_text'] = p
    diagnostic['text_with_return_interactions'] = ip
    diagnostic.to_pickle(OUT/'validation_2016_diagnostics.pkl')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
