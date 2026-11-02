"""Write a separate refined-sizing candidate without changing previous CSVs.

Uses the existing fully trained Ridge+FinBERT predictions and the multiplier
selected in research_sizing_refinement.py. Never uploads files.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import news
from research_exposure_sizing import multiplier

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    source = ROOT/'predictions_ridge_finbert.csv'
    previous = ROOT/'predictions_ridge_finbert_sized.csv'
    destination = ROOT/'predictions_ridge_finbert_sized_v2.csv'
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, previous]}
    original_audit = json.loads((ROOT/'finbert_results/final_csv_audit.json').read_text())
    assert hashes[str(source)] == original_audit['sha256']
    selected = json.loads((ROOT/'sizing_refinement_results/selection.json').read_text())
    assert selected['return_model_spec'] == {'alpha': 100., 'fb_scale': .1}
    assert selected['sizing_spec'] == {'vol_power': 1., 'liquidity_power': -1.}
    train, test = news.get_ins_data(), news.get_oos_data()
    base = pd.read_csv(source, dtype={'headline_id': 'int64', 'pred': 'float64'},
                       float_precision='round_trip')
    assert list(base.columns) == ['headline_id', 'pred']
    assert np.array_equal(base.headline_id.to_numpy(), test.headline_id.to_numpy())
    scale = multiplier(train, test, selected['sizing_spec'])
    pred = base.pred.to_numpy() * scale
    assert np.isfinite(pred).all() and ((scale >= .25) & (scale <= 4)).all()
    assert np.array_equal(np.sign(pred), np.sign(base.pred))
    news.write_submission(prediction_df=test.assign(pred=pred), path=destination)
    written = pd.read_csv(destination, dtype={'headline_id': 'int64', 'pred': 'float64'},
                          float_precision='round_trip')
    assert list(written.columns) == ['headline_id', 'pred']
    assert written.headline_id.is_unique and len(written) == len(test)
    assert np.array_equal(written.headline_id.to_numpy(), base.headline_id.to_numpy())
    assert np.array_equal(written.pred.to_numpy(), pred)
    for p in [source, previous]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    audit = {'csv': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
             'rows': len(written), 'previous_csv_hashes_unchanged': hashes,
             'selected_sizing': selected['sizing_spec'],
             'formula': 'clip((train_median_vol / floored_vol) * (train_median_weight / floored_weight), .25, 4)',
             'vol_floor': max(float(train.vol.quantile(.05)), .001),
             'weight_floor': max(float(train.weight.quantile(.05)), 1.),
             'training_median_vol': float(train.vol.median()),
             'training_median_weight': float(train.weight.median()),
             'sizing_fit_years': [2013, 2014, 2015, 2016],
             'multiplier_quantiles': pd.Series(scale).quantile([0, .01, .5, .99, 1]).to_dict(),
             'ids_order_finite_values_round_trip_verified': True, 'uploaded': False}
    (ROOT/'sizing_refinement_results/final_csv_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
