import os, sys, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.runtime'))
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np, pandas as pd, torch
from torch import nn
from common2 import load_base, folds_cache, ART, TARGETS

p = argparse.ArgumentParser()
p.add_argument('--name', default='nn5')
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--seed', type=int, default=43)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--dim', type=int, default=768)
p.add_argument('--lr', type=float, default=0.002)
args = p.parse_args()
torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
d = load_base(); x = d['x']; y = d['y']; nt = d['ntrain']
fold = folds_cache(args.folds, 42)
from features2 import build_extra, build_aggregates
from sklearn.preprocessing import QuantileTransformer
extra = build_extra(); agg = build_aggregates(fold)
cat_cols = ['microcat_name', 'subcategory_name', 'category_name', 'full_category', 'item_condition', 'title_first1', 'title_first2']
carr = []; card = []
for c in cat_cols:
    s = x[c].where(x[c].map(x[c].value_counts()) >= 3, 'missing')
    a, names = pd.factorize(s); carr.append(a); card.append(len(names))
cat = torch.tensor(np.array(carr).T, dtype=torch.long, device='cuda')
num = x.drop(columns=d['cats'] + ['title', 'description']).to_numpy(dtype=np.float32)
num = np.hstack([num, extra.to_numpy(dtype=np.float32), agg])
num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
num = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42, subsample=200000).fit_transform(num).astype(np.float32)
num = np.clip(num, -8, 8)
sem = np.load(ART / 'embeddings.npy'); sem = sem / (np.linalg.norm(sem, axis=1, keepdims=True) + 1e-6)
big = np.load(ART / 'emb_e5large.npy'); big = big / (np.linalg.norm(big, axis=1, keepdims=True) + 1e-6)
svd = np.load(ART / 'text_svd.npy')[:, :224]
svd = (svd - svd.mean(0)) / (svd.std(0) + 1e-6)
knn = np.load(ART / 'knn.npy')
knn[:, :2] = (knn[:, :2] - 0.85) * 5.0
dense = torch.tensor(np.hstack([num, sem * 1.0, big * 1.0, svd * 0.3, knn]).astype(np.float32), device='cuda')
ys = torch.tensor(y, device='cuda')
print('DENSE', dense.shape, 'cat cards', card, flush=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = nn.ModuleList([nn.Embedding(n, min(48, int(np.sqrt(n)) + 1)) for n in card])
        self.pre = nn.Sequential(nn.LazyLinear(args.dim), nn.BatchNorm1d(args.dim), nn.SiLU())
        self.body = nn.Sequential(nn.Dropout(0.15), nn.Linear(args.dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
                                  nn.Dropout(0.2), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 4))
    def forward(self, ids):
        z = torch.cat([dense[ids]] + [e(cat[ids, j]) for j, e in enumerate(self.cat)], dim=1)
        return self.body(self.pre(z))

def predict(model, ids):
    model.eval(); out = []
    with torch.inference_mode():
        for a in range(0, len(ids), 4096):
            out.append(model(ids[a:a + 4096]).float().cpu().numpy())
    return np.vstack(out)

oof = np.zeros((nt, 4), dtype=np.float32); test = np.zeros((len(x) - nt, 4), dtype=np.float32)
t0 = time.time()
for k in range(args.folds):
    itr = np.where(fold[:nt] != k)[0]; iva = np.where(fold[:nt] == k)[0]
    model = Net().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    best = 9.9; best_state = None; wait = 0
    for ep in range(args.epochs):
        model.train(); ids = np.random.permutation(itr); losses = []
        for a in range(0, len(ids), 1024):
            idx = ids[a:a + 1024]
            if len(idx) < 32: continue
            opt.zero_grad(set_to_none=True)
            pred = model(idx)
            loss = torch.abs(pred - ys[idx]).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step()
            losses.append(loss.item())
        v = predict(model, iva[:20000]); sc = np.abs(v - y[iva[:20000]]).mean()
        print('EPOCH', k, ep + 1, 'train %.4f' % np.mean(losses), 'val %.5f' % sc, 'lr %.5f' % opt.param_groups[0]['lr'], 'sec', round(time.time() - t0), flush=True)
        sched.step(sc)
        if sc < best: best = sc; best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 5: break
    model.load_state_dict(best_state)
    oof[iva] = predict(model, iva); test += predict(model, np.arange(nt, len(x))) / args.folds
    print('FOLD DONE', k, 'best %.5f' % best, flush=True)
    del model
    torch.cuda.empty_cache()
macro = np.abs(oof - y).mean()
print('MACRO %.5f SCORE %.5f %.1fs' % (macro, 1 / (1 + macro), time.time() - t0), flush=True)
np.savez_compressed(ART / (args.name + '_oof.npz'), oof=oof, test=test, want=np.array([0, 1, 2, 3]))
