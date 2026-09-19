import numpy as np, pandas as pd
from common2 import load_base, folds_cache, ART, TARGETS
d = load_base(); x = d['x']; y = d['y']; nt = d['ntrain']
z = np.load(ART / 'cat_v2_oof.npz'); oof = z['oof']
err = np.abs(oof - y)
print('per-target MAE', np.round(err.mean(0), 4), 'macro', round(err.mean(), 5))
print('bias (pred-true)', np.round((oof - y).mean(0), 4))
print('median err', np.round(np.median(err, 0), 4))
for j, t in enumerate(TARGETS):
    print(t, 'given-target MAE %.5f' % err[:, j].mean())
# error by microcat (top contributors)
mc = x['microcat_name'].astype(str).to_numpy()[:nt]
g = pd.DataFrame({'mc': mc, 'e': err.mean(1)}).groupby('mc').agg(n=('e', 'size'), e=('e', 'mean'), tot=('e', 'sum')).sort_values('tot', ascending=False)
print(g.head(15).round(3).to_string())
print('---- height error by microcat')
gh = pd.DataFrame({'mc': mc, 'e': err[:, 1], 'n': 1}).groupby('mc').agg(e=('e', 'mean'), n=('n', 'size')).sort_values('e', ascending=False)
print(gh[gh.n > 200].head(12).round(3).to_string())
# how often exact equality h==w==l in truth, and our error there
Y = np.expm1(y)
same_hw = np.abs(Y[:, 1] - Y[:, 3]) < 1e-6
same_wl = np.abs(Y[:, 3] - Y[:, 2]) < 1e-6
print('share h==w %.4f w==l %.4f' % (same_hw.mean(), same_wl.mean()))
print('MAE on h==w rows %.4f vs others %.4f' % (err[same_hw, 3].mean(), err[~same_hw, 3].mean()))
print('MAE height on w==l rows %.4f vs others %.4f' % (err[same_wl, 1].mean(), err[~same_wl, 1].mean()))
