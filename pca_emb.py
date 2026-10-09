import numpy as np, time
from sklearn.decomposition import PCA
from common2 import ART

for src, dst, k in [
    ('emb_e5large.npy', 'emb_large_pca192.npy', 192),
    ('embeddings.npy', 'emb_small_pca96.npy', 96),
    ('image_resnet50.npy', 'image_pca192.npy', 192),
    ('image_vit_b16.npy', 'image_vit_pca192.npy', 192),
]:
    t = time.time()
    a = np.load(ART / src)
    z = PCA(
        n_components=k, svd_solver='randomized', random_state=42,
    ).fit_transform(a).astype(np.float32)
    z /= (z.std(0) + 1e-6)
    np.save(ART / dst, z)
    print(src, '->', dst, z.shape, round(time.time() - t, 1), flush=True)
