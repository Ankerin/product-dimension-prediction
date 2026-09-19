import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
COLS = ['target_weight', 'target_height', 'target_length', 'target_width']


def ordered(values):
    out = np.expm1(values.copy())
    h = np.minimum(np.minimum(out[:, 1], out[:, 3]), out[:, 2])
    l = np.maximum(np.maximum(out[:, 1], out[:, 3]), out[:, 2])
    mid = out[:, 1] + out[:, 3] + out[:, 2] - h - l
    out[:, 1], out[:, 3], out[:, 2] = h, mid, l
    return out


def main():
    cat = np.load(ART / 'cat_v3_oof.npz')
    image = [np.load(ART / ('image_text_' + suffix + '_oof.npz')) for suffix in ['w', 'h', 'l', 'x']]
    image_weights = [0.43039, 0.49391, 0.48621, 0.48355]
    pred = cat['test'].copy()
    for j, (artifact, weight) in enumerate(zip(image, image_weights)):
        pred[:, j] = (1.0 - weight) * cat['test'][:, j] + weight * artifact['test'][:, j]
    values = ordered(pred)
    values[:, 0] = np.clip(values[:, 0], 0.001, 5000.0)
    values[:, 1:] = np.maximum(values[:, 1:], 1.0)
    values[:, 1:] = np.round(values[:, 1:])
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    out = pd.DataFrame({'item_id': sample.item_id.to_numpy()})
    for j, col in enumerate(COLS):
        out[col] = np.round(values[:, j], 6)
    out.to_csv(ROOT / 'submission_vision_blend.csv', index=False)
    print('wrote', out.shape, 'image_weights', image_weights, flush=True)


if __name__ == '__main__':
    main()