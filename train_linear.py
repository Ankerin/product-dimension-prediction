import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import time, json
import numpy as np
from scipy import sparse, optimize
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import prepare, ART

d = prepare()
df = d['x']
it = d['train_idx']
iv = d['valid_idx']
nt = d['ntrain']
y = d['y']

cats = ['full_category', 'item_condition', 'title_first1', 'title_first2', 'month', 'seller_id_cat']
enc = OneHotEncoder(handle_unknown='ignore', min_frequency=3, dtype=np.float32)
cat = enc.fit_transform(df[cats].astype(str))

parts = [
    sparse.load_npz(ART / (n + '.npz')) * s
    for n, s in [('title_word', 1.5), ('title_char', 1.), ('desc_word', .8)]
]

num = df[[
    'item_price_log', 'day', 'title_len', 'description_len',
    'regex_kg', 'regex_ml', 'regex_cm', 'title_num0', 'title_num1',
]].to_numpy().astype(np.float32)
num = np.clip(StandardScaler().fit_transform(num), -5, 5)

x = sparse.hstack(
    [*parts, cat, sparse.csr_matrix(num), np.ones((len(df), 1), dtype=np.float32)],
    format='csr', dtype=np.float32,
)
del parts, cat

xt = x[it]
xv = x[iv]
xe = x[nt:]

center = np.median(y[it], axis=0)
target = y[it] - center
n, p = xt.shape
print('MATRIX', x.shape, x.nnz, flush=True)

w = np.zeros((p, 4), dtype=np.float64)
counter = 0
start = time.time()
reg = 3.
eps = .04


def obj(flat):
    global counter
    w = flat.reshape(p, 4)
    e = xt @ w - target
    a = np.sqrt(e * e + eps * eps)
    loss = (a.sum() + reg * .5 * (w * w).sum()) / n
    grad = (xt.T @ (e / a) + reg * w) / n
    counter += 1
    if counter % 20 == 0:
        val = np.abs(xv @ w + center - y[iv]).mean(axis=0)
        print(
            'STEP', counter, 'loss', loss, 'mae', val,
            'score', 1 / (1 + val.mean()),
            'sec', round(time.time() - start),
            flush=True,
        )
    return loss, grad.ravel()


res = optimize.minimize(
    obj, w.ravel(), jac=True, method='L-BFGS-B',
    options={'maxiter': 160, 'maxcor': 8, 'ftol': 1e-8},
)
w = res.x.reshape(p, 4)
vp = xv @ w + center
tp = xe @ w + center

np.savez_compressed(ART / 'linear_pred.npz', valid=vp, test=tp)
np.save(ART / 'linear_weights.npy', w.astype(np.float32))

err = np.abs(vp - y[iv]).mean(axis=0)
print('FINAL', err, 1 / (1 + err.mean()), res.message, flush=True)
