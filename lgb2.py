import os

os.environ['OMP_NUM_THREADS'] = '6'

import argparse, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy import sparse
from common2 import load_base, folds_cache, ART, TARGETS

p = argparse.ArgumentParser()
p.add_argument('--name', default='lgb5')
p.add_argument('--emb2', default='emb_e5large.npy')
p.add_argument('--iters', type=int, default=4000)
p.add_argument('--lr', type=float, default=0.03)
p.add_argument('--leaves', type=int, default=127)
p.add_argument('--threads', type=int, default=6)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--tfidf', type=int, default=4000)
p.add_argument('--agg', action='store_true', default=True)
p.add_argument('--noagg', dest='agg', action='store_false')
p.add_argument('--seed', type=int, default=7)
args = p.parse_args()

d = load_base()
x = d['x'].drop(columns=['title', 'description']).copy()
y = d['y']
nt = d['ntrain']
fold = folds_cache(args.folds, 42)

from features2 import build_extra, build_aggregates

extra = build_extra()
agg = build_aggregates(fold) if args.agg else None

ext = np.load(ART / 'emb_large_pca192.npy')
small = np.load(ART / 'emb_small_pca96.npy')

parts = [
    sparse.csr_matrix(np.load(ART / 'text_svd.npy')[:, :224]),
    sparse.csr_matrix(small),
    sparse.csr_matrix(ext),
]

for name in ['title_word', 'title_char', 'desc_word']:
    a = sparse.load_npz(ART / (name + '.npz'))
    freq = np.asarray((a > 0).sum(axis=0)).ravel()
    ids = np.argsort(freq)[-args.tfidf:]
    a = a[:, ids]
    a.data[:] = 1
    parts.append(a)

cat_cols = list(d['cats'])
num = x.drop(columns=cat_cols).to_numpy(dtype=np.float32)

codes = []
for c in cat_cols:
    v = x[c].astype('category').cat.codes.to_numpy(dtype=np.float32).reshape(-1, 1)
    codes.append(v)

knn = np.load(ART / 'knn.npy')

dense = np.hstack(
    [num] + codes + [extra.to_numpy(dtype=np.float32), knn]
    + ([agg] if agg is not None else [])
)

X = sparse.hstack(parts + [sparse.csr_matrix(dense)], format='csr', dtype=np.float32)
print('MATRIX', X.shape, X.nnz, flush=True)

cat_indices = [X.shape[1] - dense.shape[1] + num.shape[1] + i for i in range(len(cat_cols))]

test_cache = {}
oof = np.zeros((nt, 4), dtype=np.float32)
test = np.zeros((X.shape[0] - nt, 4), dtype=np.float32)
recs = []
t0 = time.time()

for j in range(4):
    st = time.time()
    for k in range(args.folds):
        itr = np.where(fold[:nt] != k)[0]
        iva = np.where(fold[:nt] == k)[0]

        ds = lgb.Dataset(
            X[itr], y[itr, j],
            feature_name=[str(i) for i in range(X.shape[1])],
            categorical_feature=cat_indices, free_raw_data=True,
        )
        dv = lgb.Dataset(X[iva], y[iva, j], reference=ds)
        params = dict(
            objective='regression_l1', metric='l1', learning_rate=args.lr,
            num_leaves=args.leaves, min_data_in_leaf=40, feature_fraction=0.35,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, max_bin=63,
            num_threads=args.threads, verbose=-1, seed=args.seed + k,
        )
        m = lgb.train(
            params, ds, num_boost_round=args.iters, valid_sets=[dv],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        oof[iva, j] = m.predict(X[iva], num_iteration=m.best_iteration)
        test[:, j] += m.predict(X[nt:], num_iteration=m.best_iteration) / args.folds
        print('FOLD', TARGETS[j], k, 'best', m.best_iteration, flush=True)
        del ds, dv, m

    mae = np.abs(oof[:, j] - y[:, j]).mean()
    recs.append({'target': TARGETS[j], 'mae': float(mae), 'sec': round(time.time() - st, 1)})
    print('TARGET', TARGETS[j], 'oof %.5f' % mae, flush=True)

macro = np.abs(oof - y).mean()
print(
    'MACRO %.5f SCORE %.5f %.1fs' % (macro, 1 / (1 + macro), time.time() - t0),
    flush=True,
)

np.savez_compressed(ART / (args.name + '_oof.npz'), oof=oof, test=test, want=np.array([0, 1, 2, 3]))
recs.append({'macro': float(macro), 'sec': round(time.time() - t0, 1)})
(ART / (args.name + '_metrics.json')).write_text(json.dumps(recs, indent=2))
