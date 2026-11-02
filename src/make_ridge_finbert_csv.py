"""Refit the selected Ridge + FinBERT model and write a LOCAL submission CSV.

Prerequisite: complete both FinBERT caches with research_finbert_cache.py.
Run: python src/make_ridge_finbert_csv.py
This script never uploads anything. Hyperparameters are frozen from research.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sqlite3

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'algoarena-matplotlib'))
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import news
from research_ridge_tuning import interactions

ROOT = Path(__file__).resolve().parent
CSV = ROOT/'predictions_ridge_finbert.csv'
MODEL_PATH = ROOT/'finbert_results'/'final_ridge_finbert.joblib'
PROBABILITY_COLUMNS = ['positive', 'negative', 'neutral']


def get_sentiment(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        data = pd.read_sql_query('SELECT * FROM sentiment', db).set_index('headline')
    assert data.index.is_unique
    assert np.isfinite(data[PROBABILITY_COLUMNS].to_numpy()).all()
    assert np.allclose(data[PROBABILITY_COLUMNS].sum(axis=1), 1, atol=1e-5)
    return data


def transform(bundle, df, sentiment):
    text = bundle['vectorizer'].transform(df.headline)
    past = np.clip(df[['returns_past_hour', 'returns_past_day']].to_numpy() /
                   np.maximum(df.vol.to_numpy()[:, None], bundle['volatility_floor']), -3, 3)
    words_and_returns = interactions(text, past, .25)
    probabilities = bundle['sentiment_scaler'].transform(
        sentiment.loc[df.headline, PROBABILITY_COLUMNS].to_numpy())
    return hstack([words_and_returns, csr_matrix(probabilities * .1)], format='csr')


def fit_model(train, sentiment):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5,
                               max_features=100000, sublinear_tf=True)
    vectorizer.fit(train.headline)
    bundle = {'vectorizer': vectorizer,
              'volatility_floor': max(float(train.vol.quantile(.05)), .001),
              'sentiment_scaler': StandardScaler().fit(
                  sentiment.loc[train.headline, PROBABILITY_COLUMNS].to_numpy()),
              'ridge': Ridge(alpha=100., solver='lsqr', tol=1e-6),
              'model_source': json.loads((ROOT/'finbert_results/model_source.json').read_text())}
    bundle['ridge'].fit(transform(bundle, train, sentiment), train.returns)
    return bundle


def main():
    os.chdir(ROOT)  # Official writer resolves df_oos.pkl relative to cwd.
    selection = json.loads((ROOT/'finbert_results/selection.json').read_text())
    assert selection['selected_model'] == 'finbert_probabilities_scale_0.1'
    train = news.get_ins_data()
    test = news.get_oos_data()
    assert 'returns' not in test.columns
    assert set(train.index.get_level_values('date').year) == {2013, 2014, 2015, 2016}
    assert train.headline_id.is_unique and test.headline_id.is_unique
    ins_sentiment = get_sentiment(ROOT/'finbert_results/sentiment_cache.sqlite')
    oos_sentiment = get_sentiment(ROOT/'finbert_results/sentiment_cache_oos.sqlite')
    assert set(train.headline).issubset(ins_sentiment.index)
    assert set(test.headline).issubset(oos_sentiment.index)

    # Verify the packaged implementation reproduces the selected research model
    # before refitting it on all labeled rows. This does not tune any parameters.
    years = train.index.get_level_values('date').year
    previous = train[years < 2016]
    validation = train[years == 2016]
    check_model = fit_model(previous, ins_sentiment)
    check_pred = check_model['ridge'].predict(transform(check_model, validation, ins_sentiment))
    reference = pd.read_pickle(ROOT/'finbert_results/validation_predictions_2016.pkl')
    assert reference.index.equals(validation.index)
    reference_pred = reference[selection['selected_model']].to_numpy()
    difference = float(np.max(np.abs(check_pred-reference_pred)))
    assert difference < 1e-8, difference
    print(f'Research model reproduced; max prediction difference={difference:.3g}', flush=True)

    model = fit_model(train, ins_sentiment)
    predictions = model['ridge'].predict(transform(model, test, oos_sentiment))
    assert len(predictions) == len(test) and np.isfinite(predictions).all()
    joblib.dump(model, MODEL_PATH)
    news.write_submission(prediction_df=test.assign(pred=predictions), path=CSV)

    # Read the CSV back and validate IDs as integers, preserving their full precision.
    written = pd.read_csv(CSV, dtype={'headline_id': 'int64', 'pred': 'float64'},
                          float_precision='round_trip')
    assert list(written.columns) == ['headline_id', 'pred']
    assert len(written) == len(test) and written.headline_id.is_unique
    assert np.array_equal(written.headline_id.to_numpy(), test.headline_id.to_numpy())
    assert np.isfinite(written.pred).all() and not written.isna().any().any()
    assert np.array_equal(written.pred.to_numpy(), predictions)
    audit = {'csv': str(CSV), 'sha256': hashlib.sha256(CSV.read_bytes()).hexdigest(),
             'training_rows': len(train), 'training_years': [2013, 2014, 2015, 2016],
             'prediction_rows': len(written), 'prediction_years': sorted(set(test.index.get_level_values('date').year.tolist())),
             'selected_model': selection['selected_model'], 'ridge_alpha': 100.,
             'word_ngram_range': [1, 3], 'past_return_interaction_strength': .25,
             'finbert_feature_scale': .1, 'finbert_model': model['model_source'],
             'volatility_floor_from_training': model['volatility_floor'],
             'prediction_clipping': False, 'liquidity_filter': False,
             'weight_multiplied_into_prediction': False,
             'csv_round_trip_verified': True, 'headline_ids_and_order_verified': True,
             'research_reproduction_max_abs_prediction_difference': difference,
             'prediction_quantiles': written.pred.quantile([0, .01, .5, .99, 1]).to_dict(),
             'positive_fraction': float((written.pred > 0).mean()), 'uploaded': False}
    (ROOT/'finbert_results/final_csv_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
