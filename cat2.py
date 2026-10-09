import os

os.environ.setdefault('OMP_NUM_THREADS', '6')

import argparse, json, time
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from common2 import load_base, folds_cache, report, ART, TARGETS


def build_matrix(args, fold):
    d = load_base()
    x = d['x'].copy()
    nt = d['ntrain']
    cats = list(d['cats'])

    texts = []
    if args.text:
        texts = ['title', 'description']
    else:
        x = x.drop(columns=['title', 'description'])

    blocks = [x]

    if args.extra:
        from features2 import build_extra
        e = build_extra()
        blocks.append(e)

    if args.svd:
        z = np.load(ART / 'text_svd.npy')
        blocks.append(pd.DataFrame(z, columns=['svd_%d' % i for i in range(z.shape[1])]))

    if args.emb:
        z = np.load(ART / 'embeddings.npy')
        blocks.append(pd.DataFrame(z, columns=['emb_%d' % i for i in range(z.shape[1])]))

    if args.emb2:
        z = np.load(ART / args.emb2)
        blocks.append(pd.DataFrame(z, columns=['emb2_%d' % i for i in range(z.shape[1])]))

    if args.emb_pca:
        for fn, tag in [('emb_large_pca192.npy', 'pl'), ('emb_small_pca96.npy', 'ps')]:
            z = np.load(ART / fn)
            blocks.append(pd.DataFrame(z, columns=[tag + '_%d' % i for i in range(z.shape[1])]))

    if args.knn:
        for fi, fn in enumerate(args.knnfile.split(',')):
            z = np.load(ART / fn)
            blocks.append(pd.DataFrame(z, columns=['knn%d_%d' % (fi, i) for i in range(z.shape[1])]))

    if args.agg:
        from features2 import build_aggregates
        a = build_aggregates(fold)
        blocks.append(pd.DataFrame(a, columns=['agg_%d' % i for i in range(a.shape[1])]))

    X = pd.concat(blocks, axis=1)

    for c in X.columns:
        if c not in cats and c not in texts and X[c].dtype != np.float32:
            X[c] = X[c].astype(np.float32)

    return X, d, cats, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='cat5')
    p.add_argument('--text', action='store_true')
    p.add_argument('--svd', action='store_true')
    p.add_argument('--emb', action='store_true')
    p.add_argument('--emb2', default='')
    p.add_argument('--agg', action='store_true')
    p.add_argument('--extra', action='store_true')
    p.add_argument('--emb_pca', action='store_true')
    p.add_argument('--knn', action='store_true')
    p.add_argument('--knnfile', default='knn.npy')
    p.add_argument('--border', type=int, default=64)
    p.add_argument('--iters', type=int, default=3500)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.045)
    p.add_argument('--patience', type=int, default=250)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--loss', default='MAE')
    p.add_argument('--targets', default='0,1,2,3')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--maxfolds', type=int, default=5)
    args = p.parse_args()

    fold = folds_cache(args.folds, args.seed if args.folds == 5 else 12345)
    X, d, cats, texts = build_matrix(args, fold)

    y = d['y']
    nt = d['ntrain']
    n = len(X)
    print('MATRIX', X.shape, 'cats', len(cats), 'texts', texts, flush=True)

    opts = dict(cat_features=cats, text_features=texts)
    params = dict(
        iterations=args.iters, depth=args.depth, learning_rate=args.lr,
        loss_function=args.loss, eval_metric='RMSE', task_type='GPU', devices='0',
        thread_count=6, random_seed=args.seed, verbose=500, l2_leaf_reg=5,
        bootstrap_type='Bernoulli', subsample=0.8, allow_writing_files=False,
        gpu_ram_part=0.75, border_count=args.border, od_type='Iter',
        od_wait=args.patience, use_best_model=True, max_ctr_complexity=1,
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

    want = [int(v) for v in args.targets.split(',')]
    oof = np.zeros((nt, len(TARGETS)), dtype=np.float32)
    test = np.zeros((n - nt, len(TARGETS)), dtype=np.float32)
    recs = []
    t0 = time.time()

    for j in range(4):
        if j not in want:
            continue

        st = time.time()
        for k in range(min(args.maxfolds, args.folds)):
            itr = np.where((fold[:nt] != k))[0]
            iva = np.where((fold[:nt] == k))[0]

            model = CatBoostRegressor(**params)
            model.fit(
                Pool(X.iloc[itr], y[itr, j], **opts),
                eval_set=Pool(X.iloc[iva], y[iva, j], **opts),
            )
            oof[iva, j] = model.predict(Pool(X.iloc[iva], **opts))
            test[:, j] += model.predict(Pool(X.iloc[nt:], **opts)) / min(args.maxfolds, args.folds)

            print(
                'FOLD', TARGETS[j], k,
                'best', model.get_best_iteration(),
                'bestscore', round(model.get_best_score()['validation']['RMSE'], 5),
                flush=True,
            )
            del model

        m = np.abs(oof[:, j] - y[:, j]).mean()
        recs.append({'target': TARGETS[j], 'mae': float(m), 'sec': round(time.time() - st, 1)})
        print('TARGET', TARGETS[j], 'oof MAE %.5f' % m, flush=True)

    macro = np.abs(oof - y).mean()
    print(
        'MACRO %.5f SCORE %.5f total %.1fs'
        % (macro, 1 / (1 + macro), time.time() - t0),
        flush=True,
    )

    np.savez_compressed(ART / (args.name + '_oof.npz'), oof=oof, test=test, want=np.array(want))
    recs.append({'macro': float(macro), 'score': float(1 / (1 + macro)), 'sec': round(time.time() - t0, 1)})
    (ART / (args.name + '_metrics.json')).write_text(json.dumps(recs, indent=2))


if __name__ == '__main__':
    main()
