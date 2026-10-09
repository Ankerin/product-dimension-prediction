import os

os.environ.setdefault('OMP_NUM_THREADS', '6')

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)
TARGETS = ['real_weight', 'real_height', 'real_length', 'real_width']

HOMO = str.maketrans('aceopxykmthb', 'асеорхукмтнв')
WORD = re.compile(r'[a-zа-яё]+', re.I)


def normalize(s):
    s = str(s).lower().replace('ё', 'е')
    s = WORD.sub(
        lambda m: m[0].translate(HOMO) if re.search('[а-я]', m[0]) else m[0], s,
    )
    return re.sub(r'[^a-zа-я0-9]+', ' ', s).strip() or 'empty'


def prepare():
    cache = ART / 'features.pkl'
    if cache.exists():
        return pd.read_pickle(cache)

    tr, te = pd.read_parquet(ROOT / 'train.parquet'), pd.read_parquet(ROOT / 'test.parquet')
    df = pd.concat([tr[te.columns], te], ignore_index=True)

    x = pd.DataFrame(index=df.index)

    cats = ['item_condition', 'category_name', 'subcategory_name', 'microcat_name']
    for c in cats:
        x[c] = df[c].fillna('missing').astype(str)
    x['full_category'] = x.subcategory_name + ' / ' + x.microcat_name
    cats += ['full_category']

    for c in ['item_id', 'seller_id', 'buyer_id', 'item_price']:
        x[c] = df[c].astype(float)
        x[c + '_log'] = np.log1p(df[c].clip(lower=0))

    for c in ['seller_id', 'buyer_id', 'microcat_name']:
        x[c + '_count'] = df[c].map(df[c].value_counts()).astype(float)

    for c in ['seller_id', 'buyer_id']:
        x[c + '_cat'] = df[c].astype(str)
        cats.append(c + '_cat')

    dates = pd.to_datetime(df.order_date)
    x['day'] = dates.dt.dayofyear
    x['month'] = dates.dt.month
    x['weekday'] = dates.dt.dayofweek
    x['monthday'] = dates.dt.day

    for c in ['title', 'description']:
        raw = df[c].fillna('').astype(str)
        x[c + '_len'] = raw.str.len()
        x[c + '_digits'] = raw.str.count(r'\d')
        x[c + '_lines'] = raw.str.count('\n')
        x[c + '_words'] = raw.str.count(r'\S+')
        x[c] = raw.map(normalize)

    words = x.title.str.split()
    for n in [1, 2, 3]:
        c = 'title_first' + str(n)
        x[c] = words.map(lambda w: ' '.join(w[:n]))
        cats.append(c)

    nums = x.title.map(lambda s: [float(v) for v in re.findall(r'\b\d{1,4}\b', s)][:6])
    for n in range(4):
        x['title_num' + str(n)] = nums.map(lambda a: a[n] if len(a) > n else -1)
    x['title_num_max'] = nums.map(lambda a: max(a) if a else -1)
    x['title_num_min'] = nums.map(lambda a: min(a) if a else -1)

    raw = (df.title + ' ' + df.description).str.lower().str.replace(',', '.', regex=False)
    patterns = {
        'kg': r'(\d+(?:\.\d+)?)\s*(?:кг|kg|килограмм)',
        'g': r'(\d+(?:\.\d+)?)\s*(?:гр\b|г\b|грамм)',
        'ml': r'(\d+(?:\.\d+)?)\s*(?:мл|ml|миллилитр)',
        'cm': r'(\d+(?:\.\d+)?)\s*(?:см|cm|сантиметр)',
        'mm': r'(\d+(?:\.\d+)?)\s*(?:мм|mm|миллиметр)',
        'liters': r'(\d+(?:\.\d+)?)\s*(?:л\b|литр)',
        'pieces': r'(\d+)\s*(?:шт|штук)',
        'size': r'(?:размер|рост)\s*[:\-]?\s*(\d+(?:\.\d+)?)',
    }
    for name, pat in patterns.items():
        values = raw.str.findall(pat).map(lambda vals: [float(v) for v in vals if float(v) < 100000])
        x['regex_' + name] = values.map(lambda v: v[0] if v else -1)
        x['regex_' + name + '_max'] = values.map(lambda v: max(v) if v else -1)

    dimpat = re.compile(
        r'(\d{1,3}(?:\.\d+)?)\s*(?:см|мм|cm|mm)?\s*[xх×*]\s*'
        r'(\d{1,3}(?:\.\d+)?)\s*(?:см|мм|cm|mm)?\s*[xх×*]\s*'
        r'(\d{1,3}(?:\.\d+)?)\s*(см|мм|cm|mm)?'
    )

    def dims(s):
        m = dimpat.search(s)
        if not m:
            return [-1] * 3
        a = sorted(float(v) for v in m.groups()[:3])
        return [v / 10 if m[4] in ['мм', 'mm'] else v for v in a]

    ds = np.asarray(raw.map(dims).tolist())
    for j in range(3):
        x['dim_' + str(j)] = ds[:, j]

    y = np.log1p(tr[TARGETS].to_numpy()).astype(np.float32)
    it, iv = train_test_split(
        np.arange(len(tr)), test_size=.2, random_state=42, stratify=tr.category_name,
    )
    data = dict(x=x, y=y, cats=cats, train_idx=it, valid_idx=iv, ntrain=len(tr))
    pd.to_pickle(data, cache)
    return data


if __name__ == '__main__':
    d = prepare()
    print(d['x'].shape, d['cats'], flush=True)
