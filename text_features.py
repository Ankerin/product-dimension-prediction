import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from features import prepare, ART


if __name__ == '__main__':
    d = prepare()
    x = d['x']
    chunks = []

    configs = [
        (
            'title_word', x.title,
            dict(analyzer='word', ngram_range=(1, 2), max_features=85000, min_df=3),
            96,
        ),
        (
            'title_char', x.title,
            dict(analyzer='char_wb', ngram_range=(3, 5), max_features=110000, min_df=5),
            96,
        ),
        (
            'desc_word', x.title + ' ' + x.description.str.slice(0, 3500),
            dict(analyzer='word', ngram_range=(1, 2), max_features=100000, min_df=5),
            128,
        ),
    ]

    for name, texts, cfg, k in configs:
        start = time.time()
        print('START', name, flush=True)

        vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **cfg)
        a = vec.fit_transform(texts)
        sparse.save_npz(ART / f'{name}.npz', a)
        print(
            'VECTORIZED', name, a.shape, a.nnz,
            round(time.time() - start), flush=True,
        )

        svd = TruncatedSVD(n_components=k, n_iter=4, random_state=42)
        z = svd.fit_transform(a).astype(np.float32)
        np.save(ART / f'{name}_svd.npy', z)
        chunks.append(z)

        print(
            'DONE', name,
            'variance', svd.explained_variance_ratio_.sum(),
            'seconds', round(time.time() - start),
            flush=True,
        )

    np.save(ART / 'text_svd.npy', np.hstack(chunks))
