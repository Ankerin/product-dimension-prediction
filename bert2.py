import os, sys, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.runtime'))
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np, pandas as pd, torch
from torch import nn
from common2 import load_base, folds_cache, ART, TARGETS

p = argparse.ArgumentParser()
p.add_argument('--name', default='bert5')
p.add_argument('--model', default='cointegrated/rubert-tiny2')
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--bs', type=int, default=128)
p.add_argument('--len', type=int, default=160)
p.add_argument('--lr', type=float, default=3e-5)
p.add_argument('--folds', type=int, default=5)
args = p.parse_args()

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)

from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained(args.model)
enc = AutoModel.from_pretrained(args.model)

d = load_base()
y = d['y']
nt = d['ntrain']
fold = folds_cache(args.folds, 42)

tr = pd.read_parquet(ROOT / 'train.parquet')
te = pd.read_parquet(ROOT / 'test.parquet')

texts = (
    tr.category_name + '. ' + tr.subcategory_name + '. ' + tr.microcat_name + '. '
    + tr.title + '. ' + tr.title + '. '
    + tr.description.fillna('').str.replace(r'\s+', ' ', regex=True).str.slice(0, 700)
).tolist()
texts += (
    te.category_name + '. ' + te.subcategory_name + '. ' + te.microcat_name + '. '
    + te.title + '. ' + te.title + '. '
    + te.description.fillna('').str.replace(r'\s+', ' ', regex=True).str.slice(0, 700)
).tolist()

print('tokenizing', flush=True)
enc_tok = tok(texts, max_length=args.len, padding='max_length', truncation=True, return_tensors='np')
print('tokens', enc_tok['input_ids'].shape, flush=True)


class Reg(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(
            nn.Linear(base.config.hidden_size, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

    def forward(self, ids, mask):
        h = self.base(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(z)


dev = 'cuda'
oof = np.zeros((nt, 4), dtype=np.float32)
test = np.zeros((len(texts) - nt, 4), dtype=np.float32)
t0 = time.time()

ids_all = torch.tensor(enc_tok['input_ids'], dtype=torch.long)
mask_all = torch.tensor(enc_tok['attention_mask'], dtype=torch.long)
ys = torch.tensor(y, device=dev)

for k in range(args.folds):
    itr = np.where(fold[:nt] != k)[0]
    iva = np.where(fold[:nt] == k)[0]

    model = Reg(AutoModel.from_pretrained(args.model)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = int(np.ceil(len(itr) / args.bs)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(itr)
        tot = []
        for a in range(0, len(perm), args.bs):
            idx = perm[a:a + args.bs]
            if len(idx) < 8:
                continue
            ids = ids_all[idx].to(dev, non_blocking=True)
            mask = mask_all[idx].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(ids, mask)
            loss = (out - ys[idx]).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            sched.step()
            tot.append(loss.item())
        print('FOLD', k, 'EPOCH', ep + 1, 'loss %.4f' % np.mean(tot), 'sec', round(time.time() - t0), flush=True)

    model.eval()

    def predict(idx):
        out = []
        with torch.inference_mode():
            for a in range(0, len(idx), 512):
                ii = torch.tensor(idx[a:a + 512], dtype=torch.long)
                out.append(model(ids_all[ii].to(dev), mask_all[ii].to(dev)).float().cpu().numpy())
        return np.vstack(out)

    oof[iva] = predict(iva)
    test += predict(np.arange(nt, len(texts))) / args.folds
    m = np.abs(oof[iva] - y[iva]).mean()
    print('FOLD', k, 'DONE macro %.5f' % m, flush=True)

    del model
    torch.cuda.empty_cache()

mae = np.abs(oof - y).mean(0)
print('OOF per-target', np.round(mae, 4), 'MACRO %.5f' % mae.mean(), flush=True)
np.savez_compressed(ART / (args.name + '_oof.npz'), oof=oof, test=test, want=np.arange(4))
