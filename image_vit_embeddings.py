import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).to(device).eval()
    model.heads = torch.nn.Identity()
    transform = weights.transforms()

    train = pd.read_parquet(ROOT / 'train.parquet', columns=['image_name'])
    test = pd.read_parquet(ROOT / 'test.parquet', columns=['image_name'])

    names = [(ROOT / 'train.zip', n) for n in train.image_name]
    names += [(ROOT / 'test.zip', 'test/' + n) for n in test.image_name]

    out = np.lib.format.open_memmap(
        ART / 'image_vit_b16.npy', mode='w+', dtype='float32',
        shape=(len(names), 768),
    )

    handles = {}
    start = time.time()

    with torch.inference_mode():
        for begin in range(0, len(names), 64):
            batch = []
            for archive, name in names[begin:begin + 64]:
                if archive not in handles:
                    handles[archive] = zipfile.ZipFile(archive)
                with handles[archive].open(name) as stream:
                    batch.append(transform(Image.open(stream).convert('RGB')))
            out[begin:begin + len(batch)] = model(
                torch.stack(batch).to(device)
            ).float().cpu().numpy()
            if begin % (64 * 80) == 0:
                print(
                    'VIT', begin, '/', len(names),
                    'seconds', round(time.time() - start),
                    flush=True,
                )

    out.flush()
    for handle in handles.values():
        handle.close()

    print('DONE', out.shape, 'seconds', round(time.time() - start), flush=True)


if __name__ == '__main__':
    main()
