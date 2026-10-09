import numpy as np, pandas as pd, itertools, json
from scipy.optimize import minimize
from common2 import ROOT, ART, TARGETS, load_base
import make_sub


def load(names):
    oofs = []
    tests = []
    for n in names:
        z = np.load(ART / (n + '_oof.npz'))
        oofs.append(z['oof'])
        tests.append(z['test'])
    return np.stack(oofs), np.stack(tests)


def blend(names, out_name='blend', report=True):
    d = load_base()
    y = d['y']
    nt = d['ntrain']
    O, T = load(names)
    n = len(names)

    Wt = np.zeros((n, 4))
    for j in range(4):
        target = y[:, j]

        def obj(w):
            w = np.abs(w)
            p = np.tensordot(w, O[:, :, j], axes=1)
            e = np.abs(p - target).mean()
            return e

        best = None
        for init in [np.ones(n) / n]:
            r = minimize(
                obj, init, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-7},
            )
            if best is None or r.fun < best.fun:
                best = r

        Wt[:, j] = np.abs(best.x)

    oof_blend = np.zeros_like(O[0])
    for i in range(n):
        oof_blend += O[i] * Wt[i][None, :]

    mae = np.abs(oof_blend - y).mean(axis=0)
    macro = mae.mean()
    print(
        'BLEND per-target', np.round(mae, 5),
        'macro %.5f score %.5f' % (macro, 1 / (1 + macro)),
        flush=True,
    )
    for i, nm in enumerate(names):
        print('  w', nm, np.round(Wt[i], 3), flush=True)

    test_blend = np.zeros_like(T[0])
    for i in range(n):
        test_blend += T[i] * Wt[i][None, :]

    np.savez_compressed(
        ART / (out_name + '_oof.npz'),
        oof=oof_blend, test=test_blend, want=np.arange(4), weights=Wt,
    )
    return oof_blend, test_blend, Wt


if __name__ == '__main__':
    import sys
    names = sys.argv[1].split(',')
    out = sys.argv[2] if len(sys.argv) > 2 else 'blend'
    oof, test, Wt = blend(names, out)
    (ART / (out + '_weights.json')).write_text(
        json.dumps({'names': names, 'weights': Wt.tolist()}, indent=2)
    )
