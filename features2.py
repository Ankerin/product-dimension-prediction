import os, re, time
import numpy as np, pandas as pd
from pathlib import Path
from common2 import ROOT, ART, TARGETS, load_base

NUM = r'(\d+(?:[.,]\d+)?)'
UNITS = {
    'kg': r'(?:кг|kg|килограмм)',
    'g': r'(?:грамм|гр|г|g)\b',
    'cm': r'(?:см|cm|сантиметр)',
    'mm': r'(?:мм|mm|миллиметр)',
    'ml': r'(?:мл|ml|миллилитр)',
    'l': r'(?:литр|л|l)\b',
    'm': r'(?:метр|м|m)\b',
}
KEYWORDS = {
    'height': r'(?:высота|высотой|высоту)',
    'width': r'(?:ширина|шириной|ширину)',
    'length': r'(?:длина|длиной|длину)',
    'depth': r'(?:глубина|глубиной)',
    'thick': r'(?:толщина|толщиной|толщину)',
    'diam': r'(?:диаметр)',
    'volume': r'(?:объем|обьем)',
    'weight': r'(?:вес|весом|весит)',
    'size': r'(?:размер|габарит)',
}
DIMTRIPLE = re.compile(r'(\d{1,4}(?:[.,]\d+)?)\s*(?:см|cm|мм|mm)?\s*[xх×*]\s*(\d{1,4}(?:[.,]\d+)?)\s*(?:см|cm|мм|mm)?\s*[xх×*]\s*(\d{1,4}(?:[.,]\d+)?)\s*(см|cm|мм|mm|м|г|kg|кг)?')

def _floats(vals):
    out = []
    for v in vals:
        try:
            out.append(float(str(v).replace(',', '.')))
        except Exception:
            pass
    return out

def build_extra():
    path = ART / 'x_extra.pkl'
    if path.exists():
        return pd.read_pickle(path)
    tr = pd.read_parquet(ROOT / 'train.parquet')
    te = pd.read_parquet(ROOT / 'test.parquet')
    df = pd.concat([tr[te.columns], te], ignore_index=True)
    title = df.title.fillna('')
    desc = df.description.fillna('')
    raw = (title + ' . ' + desc).str.lower().str.replace(',', '.', regex=False)
    x = pd.DataFrame(index=df.index)
    for name, pat in UNITS.items():
        rx = re.compile(NUM + r'\s*' + pat)
        vals = raw.map(lambda s: _floats(rx.findall(s)))
        x['u_' + name + '_first'] = vals.map(lambda v: v[0] if v else -1.0).astype(np.float32)
        x['u_' + name + '_last'] = vals.map(lambda v: v[-1] if v else -1.0).astype(np.float32)
        x['u_' + name + '_max'] = vals.map(lambda v: max(v) if v else -1.0).astype(np.float32)
        x['u_' + name + '_min'] = vals.map(lambda v: min(v) if v else -1.0).astype(np.float32)
        x['u_' + name + '_cnt'] = vals.map(len).astype(np.float32)
    for name, pat in KEYWORDS.items():
        rx = re.compile(pat + r'[^0-9]{0,15}' + NUM)
        vals = raw.map(lambda s: _floats(rx.findall(s)))
        x['kw_' + name] = vals.map(lambda v: v[0] if v else -1.0).astype(np.float32)
        x['kw_' + name + '_max'] = vals.map(lambda v: max(v) if v else -1.0).astype(np.float32)
        x['has_' + name] = raw.str.contains(pat, regex=True).astype(np.int8)
    triples = raw.map(lambda s: DIMTRIPLE.findall(s))
    def dims_of(t, which):
        out = []
        for g in t:
            a = [float(g[i].replace(',', '.')) for i in range(3)]
            if g[3] in ('мм', 'mm'):
                a = [v / 10.0 for v in a]
            elif g[3] == 'м':
                a = [v * 100.0 for v in a]
            elif g[3] in ('г', 'kg', 'кг'):
                continue
            out.append(sorted(a))
        if not out:
            return -1.0
        return out[0][which]
    x['trip_cnt'] = triples.map(len).astype(np.float32)
    x['trip_min'] = triples.map(lambda t: dims_of(t, 0)).astype(np.float32)
    x['trip_mid'] = triples.map(lambda t: dims_of(t, 1)).astype(np.float32)
    x['trip_max'] = triples.map(lambda t: dims_of(t, 2)).astype(np.float32)
    x['trip_vol'] = (x[['trip_min', 'trip_mid', 'trip_max']].prod(axis=1)).where(x.trip_min > 0, -1.0).astype(np.float32)
    digits = raw.str.count(r'\d')
    x['txt_digit_ratio'] = (digits / raw.str.len().clip(lower=1)).astype(np.float32)
    x['txt_has_delivery'] = raw.str.contains(r'доставк|отправ|сдэк|почт', regex=True).astype(np.int8)
    x['txt_has_set'] = raw.str.contains(r'набор|комплект|пара\b|шт', regex=True).astype(np.int8)
    x['txt_n_paren'] = raw.str.count(r'\(').astype(np.float32)
    x['txt_n_upper'] = title.str.count(r'[A-ZА-Я]').astype(np.float32)
    price = df.item_price.astype(float).fillna(1.0).clip(lower=1)
    x['price'] = price.astype(np.float32)
    for key in ['microcat_name', 'subcategory_name', 'category_name']:
        g = df.groupby(df[key].astype(str))['item_price']
        med = df[key].astype(str).map(g.median())
        x['price_med_' + key[:5]] = med.astype(np.float32)
        x['price_ratio_' + key[:5]] = (price / med.clip(lower=1)).astype(np.float32)
        cnt = df[key].astype(str).map(df[key].astype(str).value_counts())
        x['cnt_' + key[:5]] = cnt.astype(np.float32)
    x.to_pickle(path)
    print('extra features', x.shape, flush=True)
    return x

AGG_KEYS = {
    'micro': ['microcat_name'],
    'micro_cond': ['microcat_name', 'item_condition'],
    'micro_t1': ['microcat_name', 'title_first1'],
    'subcat': ['subcategory_name'],
    'fullcat': ['full_category'],
    'seller': ['seller_id_cat'],
    'buyer': ['buyer_id_cat'],
    'micro_price': ['microcat_name', 'pricebin'],
    'cat_cond': ['category_name', 'item_condition'],
}

def build_aggregates(fold, cache='agg5.npy'):
    path = ART / cache
    if path.exists():
        return np.load(path)
    d = load_base(); x = d['x']; y = d['y']; nt = d['ntrain']
    xx = x.copy()
    xx['pricebin'] = (xx['item_price_log'] * 2).round().astype(str)
    blocks = []
    for name, cols in AGG_KEYS.items():
        key = xx[cols].astype(str).agg(' | '.join, axis=1).to_numpy()
        arr = np.full((len(x), 9), np.nan, dtype=np.float32)
        def encode(ref_idx, query_idx):
            src = pd.DataFrame(y[ref_idx], columns=['y' + str(j) for j in range(4)])
            src['key'] = key[ref_idx]
            g = src.groupby('key').agg('median')
            gstd = src.groupby('key').std()
            n = src.groupby('key').size()
            q = pd.Series(key[query_idx])
            med = g.reindex(q).to_numpy(dtype=np.float32)
            std = gstd.reindex(q).to_numpy(dtype=np.float32)
            cnt = n.reindex(q).to_numpy(dtype=np.float32).reshape(-1, 1)
            return np.hstack([med, std, cnt])
        tr_all = np.where(fold[:nt] >= 0)[0]
        nk = int(fold[tr_all].max()) + 1
        for k in range(nk):
            ref = tr_all[fold[tr_all] != k]
            q = tr_all[fold[tr_all] == k]
            arr[q] = encode(ref, q)
        te_idx = np.arange(nt, len(x))
        arr[te_idx] = encode(tr_all, te_idx)
        blocks.append(arr)
        print('AGG', name, flush=True)
    out = np.hstack(blocks)
    out = np.nan_to_num(out, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    np.save(path, out)
    return out
