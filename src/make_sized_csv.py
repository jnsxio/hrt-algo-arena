"""Create a separate local CSV with the validated exposure multiplier.

Keeps predictions_ridge_finbert.csv unchanged. No model retuning or uploads.
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
    destination = ROOT/'predictions_ridge_finbert_sized.csv'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    selected = json.loads((ROOT/'exposure_sizing_results/selection.json').read_text())
    assert selected['selected_model'] == 'inverse_vol_lower_liquidity'
    assert original_hash == selected['original_csv_sha256']
    train, test = news.get_ins_data(), news.get_oos_data()
    base = pd.read_csv(source, dtype={'headline_id': 'int64', 'pred': 'float64'},
                       float_precision='round_trip')
    assert list(base.columns) == ['headline_id', 'pred']
    assert base.headline_id.is_unique
    assert np.array_equal(base.headline_id.to_numpy(), test.headline_id.to_numpy())
    scale = multiplier(train, test, selected['spec'])
    pred = base.pred.to_numpy() * scale
    assert np.isfinite(pred).all()
    assert ((scale >= .25) & (scale <= 4)).all()
    assert np.array_equal(np.sign(pred), np.sign(base.pred))
    news.write_submission(prediction_df=test.assign(pred=pred), path=destination)
    result = pd.read_csv(destination, dtype={'headline_id': 'int64', 'pred': 'float64'},
                         float_precision='round_trip')
    assert list(result.columns) == ['headline_id', 'pred']
    assert result.headline_id.is_unique and len(result) == len(test)
    assert np.array_equal(result.headline_id.to_numpy(), base.headline_id.to_numpy())
    assert np.array_equal(result.pred.to_numpy(), pred)
    assert np.isfinite(result.pred).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    audit = {'source_csv': str(source), 'source_sha256': original_hash,
             'csv': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
             'rows': len(result), 'source_unchanged': True, 'uploaded': False,
             'model_predictions_refitted': False, 'sizing_parameters_fit_years': [2013, 2014, 2015, 2016],
             'selected_rule': selected['selected_model'], 'spec': selected['spec'],
             'multiplier_formula': 'clip((train_median_vol / max(vol, train_p05_vol, .001)) * sqrt(train_median_weight / max(weight, train_p05_weight, 1)), .25, 4)',
             'training_median_vol': float(train.vol.median()),
             'training_vol_floor': max(float(train.vol.quantile(.05)), .001),
             'training_median_weight': float(train.weight.median()),
             'training_weight_floor': max(float(train.weight.quantile(.05)), 1.),
             'scale_quantiles': pd.Series(scale).quantile([0, .01, .5, .99, 1]).to_dict(),
             'prediction_quantiles': result.pred.quantile([0, .01, .5, .99, 1]).to_dict(),
             'ids_order_finite_values_round_trip_verified': True}
    (ROOT/'exposure_sizing_results/final_csv_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
