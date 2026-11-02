"""Test whether the selected ElasticNet adds information to Ridge.

Uses saved chronological validation predictions; no fitting or OOS reads.
Selects a small convex blend on 2014/2015, checks it on reused 2016.
"""
import json
from pathlib import Path
import pandas as pd
import news
from research_baselines import metrics
from research_liquidity_bootstrap import bootstrap

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'elasticnet_results'


def main():
    selected = json.loads((OUT/'selection.json').read_text())['selected_model']
    df = news.PredictionTask(dataset_source=ROOT/'df_ins.pkl', error_manager=None,
                            label='ElasticNet blend').get_data()
    rows = []
    best = None
    for year in [2014, 2015, 2016]:
        valid = df[df.index.get_level_values('date').year == year]
        predictions = pd.read_pickle(OUT/f'validation_predictions_{year}.pkl')
        assert predictions.index.equals(valid.index)
        weights = [0., .25, .5, .75, 1.] if year < 2016 else sorted({0., best})
        daily = {}
        for w in weights:
            p = ((1-w)*predictions.baseline + w*predictions[selected]).to_numpy()
            score, pnl = metrics(valid, p)
            rows.append({'year': year, 'elasticnet_fraction': w, **score})
            daily[f'weight_{w:g}'] = pnl
        if year == 2015:
            scores = pd.DataFrame(rows).groupby('elasticnet_fraction').sharpe.mean()
            best = float(scores.idxmax())
            (OUT/'blend_selection.json').write_text(json.dumps({
                'elasticnet_fraction': best, 'selection_mean_sharpes': scores.to_dict(),
                'selection_years': [2014, 2015], 'check_year': 2016,
                'check_previously_seen': True, 'oos_accessed': False,
                'submission_written': False}, indent=2))
        if year == 2016:
            pd.DataFrame(bootstrap(pd.DataFrame(daily), 'weight_0')).to_csv(OUT/'blend_bootstrap.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'blend_metrics.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
