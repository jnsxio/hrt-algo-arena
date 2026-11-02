"""Local chronological experiments. Never reads OOS data or writes a submission.

Run from any directory: python src/research_baselines.py
2015 selects hyperparameters; 2016 is a subsequent, untuned check.
"""
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "algoarena-matplotlib"))
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import news

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_results"
ALPHAS = [1.0, 10.0, 100.0]
FEATURES = ["returns_past_hour", "returns_past_day"]


def metrics(df, pred):
    pred = np.asarray(pred)
    assert np.isfinite(pred).all()
    pnl = pd.Series(pred * df.returns.to_numpy() * df.weight.to_numpy(),
                    index=df.index.get_level_values("date")).groupby(level=0).sum()
    return {"sharpe": float(np.sqrt(252) * pnl.mean() / (pnl.std() + 1e-12)),
            "mse": float(np.mean((pred - df.returns.to_numpy()) ** 2)),
            "direction_accuracy": float(np.mean(np.sign(pred) == np.sign(df.returns))),
            "days": len(pnl)}, pnl


def fit_split(train, valid, alphas):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                               max_features=100000, sublinear_tf=True)
    xt = vectorizer.fit_transform(train.headline.fillna(""))
    xv = vectorizer.transform(valid.headline.fillna(""))
    # Bounds and scaling learned exclusively from the training period.
    lo, hi = train[FEATURES].quantile(.005), train[FEATURES].quantile(.995)
    scaler = StandardScaler()
    nt = scaler.fit_transform(train[FEATURES].clip(lo, hi, axis=1))
    nv = scaler.transform(valid[FEATURES].clip(lo, hi, axis=1))
    outputs = {}
    for name, pred in {
        "constant_long": np.ones(len(valid)),
        "fade_hour": -valid.returns_past_hour.to_numpy(),
        "fade_day": -valid.returns_past_day.to_numpy(),
        "fade_hour_clipped": -valid.returns_past_hour.clip(lo.iloc[0], hi.iloc[0]).to_numpy(),
        "fade_day_clipped": -valid.returns_past_day.clip(lo.iloc[1], hi.iloc[1]).to_numpy(),
    }.items():
        outputs[name] = (pred, {})
    designs = {"text": (xt, xv), "past_returns": (nt, nv),
               "text_plus_past_returns": (hstack([xt, csr_matrix(nt)]).tocsr(),
                                           hstack([xv, csr_matrix(nv)]).tocsr())}
    for family, (a, b) in designs.items():
        for alpha in alphas[family]:
            model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6)
            model.fit(a, train.returns)
            info = {"alpha": alpha}
            if family == "past_returns":
                info["raw_coefficients"] = dict(zip(FEATURES, (model.coef_ / scaler.scale_).tolist()))
                info["intercept"] = float(model.intercept_ - np.dot(model.coef_ / scaler.scale_, scaler.mean_))
            outputs[f"{family}_alpha_{alpha:g}"] = (model.predict(b), info)
    print(f"Fit {len(train):,} rows; validate {len(valid):,}; vocabulary {xt.shape[1]:,}", flush=True)
    return outputs


def main():
    OUT.mkdir(exist_ok=True)
    df = news.PredictionTask(dataset_source=ROOT / "df_ins.pkl", error_manager=None,
                             label="research").get_data()
    years = df.index.get_level_values("date").year
    summary = {"rows_by_year": {str(k): int(v) for k, v in pd.Series(years).value_counts().sort_index().items()},
               "selection_year": 2015, "check_year": 2016,
               "oos_accessed": False, "features": FEATURES,
               "duplicate_checks": {}}
    records, daily = [], {}
    selected = None
    for year in [2015, 2016]:
        train, valid = df[years < year], df[years == year]
        summary["duplicate_checks"][str(year)] = {
            "validation_headline_seen_in_train_fraction": float(valid.headline.isin(set(train.headline)).mean()),
            "headline_id_overlap": len(set(train.headline_id) & set(valid.headline_id))}
        grid = {k: ALPHAS for k in ["text", "past_returns", "text_plus_past_returns"]} if selected is None else {k: [v] for k, v in selected.items()}
        outputs = fit_split(train, valid, grid)
        for name, (pred, info) in outputs.items():
            m, pnl = metrics(valid, pred)
            records.append({"year": year, "model": name, **m, **info})
            daily[f"{year}_{name}"] = pnl
            print(f"{year} {name:38s} Sharpe={m['sharpe']:8.3f}  MSE={m['mse']:.6f}", flush=True)
        if selected is None:
            selected = {}
            for family in grid:
                candidates = [r for r in records if r["model"].startswith(family + "_alpha_")]
                selected[family] = max(candidates, key=lambda r: r["sharpe"])["alpha"]
            summary["selected_alphas"] = selected
            # Save choices before observing 2016.
            (OUT / "experiment_details.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(records).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "daily_pnl.csv")
    (OUT / "experiment_details.json").write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, year in zip(axes, [2015, 2016]):
        for family, alpha in selected.items():
            pnl = daily[f"{year}_{family}_alpha_{alpha:g}"]
            (pnl.cumsum() / (pnl.std() + 1e-12)).plot(ax=ax, label=family)
        ax.set_title(f"{year}: cumulative P&L / daily P&L standard deviation")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "validation_pnl.png", dpi=150)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
