import os

os.environ['OMP_NUM_THREADS'] = '6'

import argparse, json, time
import numpy as np
from catboost import CatBoostRegressor, Pool
from features import prepare, ART, TARGETS

p = argparse.ArgumentParser()
p.add_argument('--name', default='baseline')
p.add_argument('--text', action='store_true')
p.add_argument('--iters', type=int, default=2000)
p.add_argument('--depth', type=int, default=8)
p.add_argument('--lr', type=float, default=.05)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--full', action='store_true')
p.add_argument('--svd', action='store_true')
p.add_argument('--targets', default='0,1,2,3')
p.add_argument('--emb', action='store_true')
p.add_argument('--agg', action='store_true')
args = p.parse_args()

d = prepare()
x = d['x']
y = d['y']
nt = d['ntrain']

if not args.text:
    x = x.drop(columns=['title', 'description'])

if args.svd:
    import pandas as pd
    svd = np.load(ART / 'text_svd.npy')
    x = pd.concat(
        [x, pd.DataFrame(svd, columns=['svd_' + str(i) for i in range(svd.shape[1])])],
        axis=1,
    )

if args.emb:
    import pandas as pd
    emb = np.load(ART / 'embeddings.npy')
    x = pd.concat(
        [x, pd.DataFrame(emb, columns=['emb_' + str(i) for i in range(emb.shape[1])])],
        axis=1,
    )

if args.agg:
    import pandas as pd
    from aggregate_features import make_aggregates
    a = make_aggregates(args.full)
    x = pd.concat(
        [x, pd.DataFrame(a, columns=['agg_' + str(i) for i in range(a.shape[1])])],
        axis=1,
    )

it = d['train_idx']
iv = d['valid_idx']
if args.full:
    it = np.arange(nt)

params = dict(
    iterations=args.iters, depth=args.depth, learning_rate=args.lr,
    loss_function='MAE', eval_metric='MAE', task_type='GPU', devices='0',
    thread_count=6, random_seed=args.seed, verbose=250, l2_leaf_reg=5,
    bootstrap_type='Bernoulli', subsample=.8, allow_writing_files=False,
    gpu_ram_part=.78,
)

if args.text:
    params['text_processing'] = {
        'tokenizers': [{
            'tokenizer_id': 'Space',
            'separator_type': 'ByDelimiter',
            'delimiter': ' ',
        }],
        'dictionaries': [{
            'dictionary_id': 'Word',
            'max_dictionary_size': '25000',
            'occurrence_lower_bound': '5',
            'gram_order': '1',
        }],
        'feature_processing': {
            'default': [{
                'dictionaries_names': ['Word'],
                'feature_calcers': ['BoW'],
                'tokenizers_names': ['Space'],
            }],
        },
    }

opts = dict(
    cat_features=d['cats'],
    text_features=['title', 'description'] if args.text else [],
)

vp = []
tp = []
records = []

for j, t in enumerate(TARGETS):
    if str(j) not in args.targets.split(','):
        continue

    print('START', args.name, t, flush=True)
    start = time.time()

    model = CatBoostRegressor(**params)
    train = Pool(x.iloc[it], y[it, j], **opts)
    valid = Pool(x.iloc[iv], y[iv, j], **opts)
    model.fit(
        train,
        eval_set=None if args.full else valid,
        early_stopping_rounds=None if args.full else 180,
    )

    v = model.predict(valid)
    pred = model.predict(Pool(x.iloc[nt:], **opts))
    vp.append(v)
    tp.append(pred)

    err = float(np.mean(np.abs(v - y[iv, j])))
    records.append({
        'target': t,
        'mae': err,
        'iterations': int(model.tree_count_),
        'seconds': time.time() - start,
    })
    print(records[-1], flush=True)

    model.save_model(str(ART / f'{args.name}_{j}.cbm'))
    np.savez_compressed(ART / f'{args.name}_pred.npz', valid=np.array(vp).T, test=np.array(tp).T)
    (ART / f'{args.name}_metrics.json').write_text(json.dumps(records, indent=2))

err = float(np.mean(
    np.abs(np.array(vp).T - y[iv][:, [int(v) for v in args.targets.split(',')]])
))
print('MACRO', err, 'SCORE', 1 / (1 + err), flush=True)
