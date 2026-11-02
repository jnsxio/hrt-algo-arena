"""Download frozen FinBERT or compute resumable LOCAL sentiment features.

Defaults to df_ins.pkl. --dataset oos explicitly prepares separate OOS features.
No headlines are sent to an inference service; this script writes no CSV.
Usage: python research_finbert_cache.py --download
       python research_finbert_cache.py [--batch-size 64]
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import sqlite3
import time

os.environ.setdefault('HF_HOME', os.path.join(tempfile.gettempdir(), 'algoarena-huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'finbert_results'
MODEL = Path(os.environ.get('FINBERT_DIR', os.path.join(tempfile.gettempdir(), 'algoarena-finbert-model')))
MODEL_ID = 'ProsusAI/finbert'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dataset', choices=['ins', 'oos'], default='ins')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.download:
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(MODEL_ID).sha
        snapshot_download(MODEL_ID, revision=revision, local_dir=str(MODEL),
                          allow_patterns=['config.json', 'tokenizer_config.json',
                                          'special_tokens_map.json', 'vocab.txt',
                                          'tokenizer.json', 'pytorch_model.bin'])
        (OUT/'model_source.json').write_text(json.dumps({'model': MODEL_ID, 'revision': revision}, indent=2))
        print('Downloaded model', revision, flush=True)
        return
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import news
    torch.set_num_threads(4)
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True)
    model.eval().to(device)
    label_order = [model.config.id2label[i].lower() for i in range(3)]
    assert set(label_order) == {'positive', 'negative', 'neutral'}, label_order
    order = [label_order.index(x) for x in ['positive', 'negative', 'neutral']]
    suffix = '' if args.dataset == 'ins' else '_oos'
    db = sqlite3.connect(OUT/f'sentiment_cache{suffix}.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS sentiment (headline TEXT PRIMARY KEY, positive REAL, negative REAL, neutral REAL)')
    df = news.PredictionTask(dataset_source=ROOT/f'df_{args.dataset}.pkl', error_manager=None, label='FinBERT features').get_data()
    unique = df.headline.drop_duplicates().tolist()
    cached = {row[0] for row in db.execute('SELECT headline FROM sentiment')}
    pending = [h for h in unique if h not in cached]
    pending.sort(key=len)
    if args.limit:
        pending = pending[:args.limit]
    print(f'Device={device}; unique={len(unique)}; cached={len(cached)}; pending={len(pending)}', flush=True)
    # Tokenization is frozen; no vocabulary fitting or validation labels are used.
    tokens = tokenizer(pending, truncation=True, max_length=128, padding=False) if pending else {}
    truncated = 0
    start = time.monotonic()
    last_print = start
    with torch.inference_mode():
        for i in range(0, len(pending), args.batch_size):
            end = min(i + args.batch_size, len(pending))
            batch = {k: v[i:end] for k, v in tokens.items()}
            inputs = tokenizer.pad(batch, padding=True, return_tensors='pt').to(device)
            p = model(**inputs).logits.softmax(dim=-1).cpu().numpy()[:, order]
            assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-5)
            db.executemany('INSERT OR REPLACE INTO sentiment VALUES (?, ?, ?, ?)',
                           [(h, float(a), float(b), float(c)) for h, (a, b, c) in zip(pending[i:end], p)])
            db.commit()
            truncated += sum(len(v) >= 128 for v in batch['input_ids'])
            now = time.monotonic()
            if now - last_print > 20 or end == len(pending):
                print(f'{end}/{len(pending)} inferred; {end / max(now-start, .01):.1f} headlines/sec; '
                      f'elapsed {(now-start)/60:.1f} min', flush=True)
                last_print = now
    table = pd.read_sql_query('SELECT * FROM sentiment', db).set_index('headline')
    complete = set(unique).issubset(table.index)
    metadata = {'device': device, 'label_order': label_order, 'unique_headlines': len(unique),
                'cached_headlines': len(table), 'complete': complete,
                'max_token_length': 128, 'at_token_limit_this_run': truncated,
                'batch_size': args.batch_size, 'torch': torch.__version__,
                'dataset': args.dataset,
                'oos_accessed': args.dataset == 'oos', 'external_inference': False}
    (OUT/f'cache_metadata{suffix}.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
