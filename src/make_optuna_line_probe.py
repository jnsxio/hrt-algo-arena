"""Prepare one local interpolation/extrapolation probe; never upload.

t=0 is the 4.838 candidate; t=.5 is its parameter midpoint with Optuna;
t=1 is the extended Optuna candidate. Intended for a few sequential probes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import news
from research_optuna_sprint import prepare, predictions, evaluate, MODEL

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', required=True, type=float)
    args = parser.parse_args()
    assert 0 <= args.t <= 2
    os.chdir(ROOT)
    selection = json.loads((ROOT/'optuna_extended_results/selection.json').read_text())['parameters']
    a,b = selection['incumbent_v2'],selection['optuna_extended']
    p = {k: a[k]+args.t*(b[k]-a[k]) for k in a}
    assert 0 < p['lower'] < p['upper'] <= 16 and p['offset'] == 0
    df,test = news.get_ins_data(),news.get_oos_data()
    density = float(test.groupby(level='date').size().mean())
    years = df.index.get_level_values('date').year
    report = []
    for year in [2014,2015,2016]:
        train,valid = df[years<year],df[years==year]
        raw = pd.read_pickle(ROOT/'finbert_results'/f'validation_predictions_{year}.pkl')
        assert raw.index.equals(valid.index)
        data = prepare(train,valid,raw[MODEL].to_numpy(),density)
        full,sparse = evaluate(data,p)
        report.append({'year':year,'sharpe':full,'density_sharpe':sparse})
    source = pd.read_csv(ROOT/'predictions_ridge_finbert.csv',dtype={'headline_id':'int64','pred':'float64'},float_precision='round_trip')
    assert np.array_equal(source.headline_id.to_numpy(),test.headline_id.to_numpy())
    target = prepare(df,test,source.pred.to_numpy(),density)
    pred = predictions(target,p)
    assert np.isfinite(pred).all()
    filename = f'predictions_optuna_line_{args.t:g}.csv'
    destination = ROOT/filename
    news.write_submission(prediction_df=test.assign(pred=pred),path=destination)
    result = pd.read_csv(destination,dtype={'headline_id':'int64','pred':'float64'},float_precision='round_trip')
    assert np.array_equal(result.headline_id.to_numpy(),test.headline_id.to_numpy())
    assert result.headline_id.is_unique and np.array_equal(result.pred.to_numpy(),pred)
    audit = {'path':str(destination),'t':args.t,'parameters':p,'local_validation':report,
             'rows':len(result),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
             'uploaded':False,'check_year_previously_seen':True}
    (ROOT/'optuna_extended_results'/f'line_{args.t:g}_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
