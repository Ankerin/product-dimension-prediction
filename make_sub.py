import numpy as np, pandas as pd
from common2 import ROOT, ART, TARGETS

COLS = ['target_weight', 'target_height', 'target_length', 'target_width']

def to_submission(test_log, name='submission.csv', round_dims=False):
    ss = pd.read_csv(ROOT / 'sample_submission.csv')
    out = pd.DataFrame({'item_id': ss.item_id.values})
    vals = np.expm1(test_log)
    vals[:, 0] = np.clip(vals[:, 0], 0.001, 5000.0)
    h = np.minimum(np.minimum(vals[:, 1], vals[:, 3]), vals[:, 2])
    l = np.maximum(np.maximum(vals[:, 1], vals[:, 3]), vals[:, 2])
    m = vals[:, 1] + vals[:, 3] + vals[:, 2] - h - l
    vals[:, 1], vals[:, 3], vals[:, 2] = h, m, l
    if round_dims:
        vals[:, 1:] = np.round(vals[:, 1:])
    for j, c in enumerate(COLS):
        out[c] = np.round(vals[:, j], 6)
    assert list(out.item_id.values) == list(ss.item_id.values)
    out.to_csv(ROOT / name, index=False)
    print('wrote', name, out.shape, flush=True)
    print(out.describe().round(3).to_string(), flush=True)
    return out

if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'cat_v2'
    out_name = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
    round_dims = len(sys.argv) > 3 and sys.argv[3] == 'round_dims'
    path = ART / (name + '_oof.npz')
    if not path.exists():
        path = ART / (name + '_pred.npz')
    z = np.load(path)
    to_submission(z['test'], out_name, round_dims=round_dims)
