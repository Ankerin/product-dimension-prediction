import os, sys, time, json
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
TARGETS = ['real_weight','real_height','real_length','real_width']

def load_base():
    from features import prepare
    return prepare()

def folds_cache(n_splits=5, seed=42):
    path = ART / f'folds{n_splits}_{seed}.npz'
    if path.exists():
        return np.load(path)['fold']
    d = load_base(); x = d['x']; nt = d['ntrain']
    from sklearn.model_selection import StratifiedKFold
    strat = x['microcat_name'].astype(str).to_numpy()[:nt]
    fold = np.full(len(x), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (a, b) in enumerate(skf.split(np.zeros(nt), strat)):
        fold[b] = k
    np.savez_compressed(path, fold=fold)
    return fold

def metric(pred, truth):
    return np.abs(pred - truth).mean(axis=0)

def score(mae_vec):
    return 1.0 / (1.0 + float(np.mean(mae_vec)))

def report(name, pred, truth):
    m = metric(pred, truth)
    print(f'{name}: per-target {np.round(m,4)} macro {m.mean():.5f} score {score(m):.5f}', flush=True)
    return m
