"""Time-capped joint model + sizing search, followed by a separate local CSV.

Uses 2014/2015 for selection, then reports the reused 2016 check. No uploads.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import optuna
import news
from make_ridge_finbert_csv import get_sentiment
from research_ridge_tuning import interactions
from research_optuna_sprint import prepare, predictions, evaluate
from research_optuna_extended import resample_masks, objective_value
from research_baselines import metrics
from research_liquidity_bootstrap import bootstrap

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'optuna_joint_results'
COLS=['positive','negative','neutral']


def state(train, valid, sentiment_train, sentiment_valid):
    v=TfidfVectorizer(ngram_range=(1,3),min_df=5,max_features=100000,sublinear_tf=True)
    a=v.fit_transform(train.headline);b=v.transform(valid.headline)
    floor=max(float(train.vol.quantile(.05)),.001)
    def past(d):
        return np.clip(d[['returns_past_hour','returns_past_day']].to_numpy()/np.maximum(d.vol.to_numpy()[:,None],floor),-3,3)
    scaler=StandardScaler().fit(sentiment_train.loc[train.headline,COLS].to_numpy())
    st=scaler.transform(sentiment_train.loc[train.headline,COLS].to_numpy())
    sv=scaler.transform(sentiment_valid.loc[valid.headline,COLS].to_numpy())
    return {'a':a,'b':b,'zt':past(train),'zv':past(valid),'st':st,'sv':sv,
            'y':train.returns.to_numpy(),'vectorizer':v,'scaler':scaler,'vol_floor':floor}


def fit_predict(s,p):
    a=interactions(s['a'],s['zt'],p['return_strength'])
    b=interactions(s['b'],s['zv'],p['return_strength'])
    a=hstack([a,csr_matrix(s['st']*p['fb_scale'])],format='csr')
    b=hstack([b,csr_matrix(s['sv']*p['fb_scale'])],format='csr')
    model=Ridge(alpha=p['alpha'],solver='lsqr',tol=1e-6).fit(a,s['y'])
    return model.predict(b), model


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True);os.chdir(ROOT)
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('predictions_*.csv')}
    df,test=news.get_ins_data(),news.get_oos_data();years=df.index.get_level_values('date').year
    sentiment=get_sentiment(ROOT/'finbert_results/sentiment_cache.sqlite')
    density=float(test.groupby(level='date').size().mean())
    states={};risk={}
    for year in [2014,2015]:
        tr,va=df[years<year],df[years==year]
        states[year]=state(tr,va,sentiment,sentiment)
        risk[year]=prepare(tr,va,np.zeros(len(va)),density)
        resample_masks(risk[year],32,density,year+77)
    old=json.loads((ROOT/'optuna_extended_results/selection.json').read_text())['parameters']['optuna_extended']
    seed=dict(old,alpha=100.,fb_scale=.1,return_strength=.25)
    sampler=optuna.samplers.TPESampler(seed=98,n_startup_trials=16)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='maximize',sampler=sampler)
    study.enqueue_trial(seed)
    for alpha in [60.,180.,300.]:study.enqueue_trial(dict(seed,alpha=alpha))
    cache={}
    def objective(trial):
        p={'alpha':trial.suggest_float('alpha',30.,600.,log=True),
           'fb_scale':trial.suggest_categorical('fb_scale',[0.,.03,.1,.3]),
           'return_strength':trial.suggest_categorical('return_strength',[0.,.125,.25,.5]),
           'vol_power':trial.suggest_float('vol_power',.75,1.75),
           'liquidity_power':trial.suggest_float('liquidity_power',-1.6,-.8),
           'lower':trial.suggest_float('lower',.12,.3),
           'upper':trial.suggest_float('upper',4.,12.),
           'offset':trial.suggest_float('offset',-.0005,.0005)}
        saved={};scores=[]
        for year,s in states.items():
            pred,_=fit_predict(s,p);saved[year]=pred;risk[year]['raw']=pred
            scores.append(evaluate(risk[year],p))
        cache[trial.number]=saved
        value=objective_value(scores)
        trial.set_user_attr('year_scores',scores)
        if trial.number%5==0:print(f'Trial {trial.number}, objective {value:.4f}, elapsed {time.monotonic()-start:.1f}s',flush=True)
        return value
    study.optimize(objective,n_trials=80,timeout=130.)
    study.trials_dataframe().to_csv(OUT/'trials.csv',index=False)
    top=sorted([t for t in study.trials if t.value is not None],key=lambda t:t.value,reverse=True)[:5]
    for year in risk:resample_masks(risk[year],256,density,year+9900)
    checked=[]
    for t in top:
        scores=[]
        for year in risk:
            risk[year]['raw']=cache[t.number][year];scores.append(evaluate(risk[year],t.params))
        checked.append({'trial':t.number,'objective':objective_value(scores),'scores':scores,'params':t.params})
    winner=max(checked,key=lambda x:x['objective']);p=winner['params']
    selected={'trials':len(study.trials),'selected':winner,'fresh_checks':checked,
              'selection_years':[2014,2015],'check_year':2016,'check_previously_seen':True,'uploaded':False}
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2))
    print('Selected',json.dumps(winner),flush=True)
    tr,va=df[years<2016],df[years==2016];s=state(tr,va,sentiment,sentiment)
    raw,_=fit_predict(s,p);d=prepare(tr,va,raw,density)
    pred=predictions(d,p);scores,pnl=metrics(va,pred)
    previous=pd.read_csv(ROOT/'optuna_extended_results/daily_pnl_2016.csv',index_col=0,parse_dates=True)['optuna_extended']
    check=pd.DataFrame({'previous_optuna':previous,'joint_optuna':pnl})
    assert not check.isna().any().any()
    boot=pd.DataFrame(bootstrap(check,'previous_optuna',repetitions=3000))
    boot.to_csv(OUT/'bootstrap.csv',index=False)
    selected['2016_metrics']=scores;(OUT/'selection.json').write_text(json.dumps(selected,indent=2))
    print('2016 check',scores,flush=True)
    # Refit once on all labeled data and write a new candidate only.
    oos_sentiment=get_sentiment(ROOT/'finbert_results/sentiment_cache_oos.sqlite')
    s=state(df,test,sentiment,oos_sentiment);raw,model=fit_predict(s,p)
    target=prepare(df,test,raw,density);pred=predictions(target,p)
    assert np.isfinite(pred).all()
    destination=ROOT/'predictions_optuna_joint.csv'
    news.write_submission(prediction_df=test.assign(pred=pred),path=destination)
    result=pd.read_csv(destination,dtype={'headline_id':'int64','pred':'float64'},float_precision='round_trip')
    assert list(result.columns)==['headline_id','pred'] and result.headline_id.is_unique
    assert np.array_equal(result.headline_id.to_numpy(),test.headline_id.to_numpy())
    assert np.array_equal(result.pred.to_numpy(),pred)
    joblib.dump({'vectorizer':s['vectorizer'],'scaler':s['scaler'],'vol_floor':s['vol_floor'],'model':model,'parameters':p},OUT/'final_model.joblib')
    for path,digest in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (OUT/'manifest.json').write_text(json.dumps({'path':str(destination),'rows':len(result),
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'uploaded':False,'parameters':p},indent=2))
    print(boot.query('block_days==10').to_string(index=False),flush=True)
    print('Total seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
