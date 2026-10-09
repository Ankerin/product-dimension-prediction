from common2 import folds_cache
from features2 import build_extra, build_aggregates
import time

t = time.time()
fold = folds_cache(5, 42)
print(
    'folds', fold[:20],
    'counts', [int((fold[:251092] == k).sum()) for k in range(5)],
    flush=True,
)

a = build_aggregates(fold)
print('agg done', a.shape, round(time.time() - t, 1), flush=True)
