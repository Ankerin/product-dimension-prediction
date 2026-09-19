import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet50(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()

    train = pd.read_parquet(ROOT / 'train.parquet', columns=['image_name'])
    test = pd.read_parquet(ROOT / 'test.parquet', columns=['image_name'])
    names = [
        (ROOT / 'train.zip', name) for name in train.image_name
    ] + [
        (ROOT / 'test.zip', 'test/' + name) for name in test.image_name
    ]
    out = np.lib.format.open_memmap(
        ART / 'image_resnet50.npy', mode='w+', dtype='float32',
        shape=(len(names), 2048)
    )
    handles = {}
    batch_size = 128
    start = time.time()
    with torch.inference_mode():
        for begin in range(0, len(names), batch_size):
            batch = []
            for archive, name in names[begin:begin + batch_size]:
                if archive not in handles:
                    handles[archive] = zipfile.ZipFile(archive)
                with handles[archive].open(name) as stream:
                    batch.append(transform(Image.open(stream).convert('RGB')))
            x = torch.stack(batch).to(device)
            out[begin:begin + len(batch)] = model(x).float().cpu().numpy()
            if begin % (batch_size * 40) == 0:
                print('IMAGE', begin, '/', len(names), 'seconds', round(time.time() - start), flush=True)
    out.flush()
    for handle in handles.values():
        handle.close()
    print('DONE', out.shape, 'seconds', round(time.time() - start), flush=True)


if __name__ == '__main__':
    main()