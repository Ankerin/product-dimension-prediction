import os

os.environ['OMP_NUM_THREADS'] = '4'

import argparse, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from features import prepare, ART, TARGETS

p = argparse.ArgumentParser()
p.add_argument('--name', default='lgbm')
p.add_argument('--iters', type=int, default=2000)
p.add_argument('--full', action='store_true')
p.add_argument('--emb', action='store_true')
p.add_argument('--no-sparse', action='store_true')
args = p.parse_args()

d = prepare()
df = d['x'].drop(columns=['title', 'description']).copy()
nt = d['ntrain']
it = d['train_idx']
iv = d['valid_idx']
y = d['y']

if args.full:
    it = np.arange(nt)

catinds = []
for c in d['cats']:
    df[c] = df[c].astype('category').cat.codes
    catinds.append(df.columns.get_loc(c))

parts = [sparse.csr_matrix(df.to_numpy(dtype=np.float32))]

if args.emb:
    parts.append(sparse.csr_matrix(np.load(ART / 'embeddings.npy')))
else:
    parts.append(sparse.csr_matrix(np.load(ART / 'text_svd.npy')[:, :192]))

if not args.no_sparse:
    for name, nf in [('title_word', 16000), ('title_char', 16000), ('desc_word', 12000)]:
        a = sparse.load_npz(ART / (name + '.npz'))
        freq = np.asarray((a > 0).sum(axis=0)).ravel()
        ids = np.argsort(freq)[-nf:]
        a = a[:, ids]
        a.data[:] = 1
        parts.append(a)

x = sparse.hstack(parts, format='csr', dtype=np.float32)
del parts, df
print('MATRIX', x.shape, x.nnz, flush=True)

xt = x[it]
xv = x[iv]
xe = x[nt:]
vp = []
tp = []
rec = []

for j, t in enumerate(TARGETS):
    start = time.time()
    print('START', t, flush=True)

    model = lgb.LGBMRegressor(
        objective='regression_l1', n_estimators=args.iters, learning_rate=.045,
        num_leaves=31, max_bin=63, min_child_samples=50, colsample_bytree=.8,
        subsample=.85, subsample_freq=1, reg_lambda=3, n_jobs=4,
        verbosity=-1, random_state=47,
    )
    model.fit(
        xt, y[it, j], categorical_feature=catinds,
        eval_set=None if args.full else [(xv, y[iv, j])],
        callbacks=[lgb.log_evaluation(250)] + ([] if args.full else [lgb.early_stopping(120)]),
    )

    v = model.predict(xv)
    pred = model.predict(xe)
    vp.append(v)
    tp.append(pred)

    rec.append(dict(
        target=t,
        mae=float(np.abs(v - y[iv, j]).mean()),
        iterations=model.best_iteration_ or args.iters,
        seconds=time.time() - start,
    ))
    print(rec[-1], flush=True)

    model.booster_.save_model(str(ART / f'{args.name}_{j}.txt'))
    np.savez_compressed(ART / f'{args.name}_pred.npz', valid=np.array(vp).T, test=np.array(tp).T)
    (ART / f'{args.name}_metrics.json').write_text(json.dumps(rec, indent=2))

print('SCORE', 1 / (1 + np.abs(np.array(vp).T - y[iv]).mean()), flush=True)
