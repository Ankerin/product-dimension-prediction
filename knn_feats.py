import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.runtime'))

import numpy as np, pandas as pd, torch


def run(k_list=(8, 32), chunk=2048, fn='emb_e5large.npy', out='knn.npy', norm=True):
    t0 = time.time()

    E = np.load(ROOT / 'artifacts' / fn)
    if norm:
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    else:
        E = E / (E.std() + 1e-9)

    d = pd.read_pickle(ROOT / 'artifacts' / 'features.pkl')
    y = d['y']
    nt = d['ntrain']
    n = E.shape[0]

    pool = torch.tensor(E[:nt], dtype=torch.float16, device='cuda')
    Y = torch.tensor(y, dtype=torch.float16, device='cuda')
    kmax = max(k_list)

    feats = []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Q = torch.tensor(E[a:b], dtype=torch.float16, device='cuda')
        sims = Q @ pool.T

        for i in range(a, b):
            if i < nt:
                sims[i - a, i] = -2.0

        vals, idx = torch.topk(sims, kmax, dim=1)
        del sims

        block = [vals[:, 0].float().unsqueeze(1), vals[:, kmax - 1].float().unsqueeze(1)]
        for k in k_list:
            v = vals[:, :k]
            ix = idx[:, :k]
            nb = Y[ix].float()
            block += [nb.mean(1), nb.median(1).values, nb.std(1)]
            block.append(v.mean(1).float().unsqueeze(1))

        feats.append(torch.cat(block, dim=1).cpu().numpy())
        del Q, vals, idx

        if (a // chunk) % 20 == 0:
            print('KNN', a, '/', n, round(time.time() - t0), flush=True)

    F = np.vstack(feats).astype(np.float32)
    np.save(ROOT / 'artifacts' / out, F)
    print('DONE', F.shape, time.time() - t0, flush=True)


if __name__ == '__main__':
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else 'emb_e5large.npy'
    out = sys.argv[2] if len(sys.argv) > 2 else 'knn.npy'
    run(fn=fn, out=out)
