import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from features import prepare,ART

def make_aggregates(full=False):
    dest=ART/('aggregates_full.npy' if full else 'aggregates.npy')
    if dest.exists():return np.load(dest)
    d=prepare();x=d['x'].copy();nt=d['ntrain'];y=d['y'];it=np.arange(nt) if full else d['train_idx']
    x['pricebin']=(x.item_price_log*2).round().astype(str)
    groups=[['seller_id_cat'],['buyer_id_cat'],['seller_id_cat','full_category'],['full_category','title_first2'],['full_category','title_first3'],['full_category','item_condition'],['full_category','pricebin']]
    vals=[]
    for cols in groups:
        key=x[cols].astype(str).agg(' | '.join,axis=1)
        src=pd.DataFrame(y,columns=['y'+str(j) for j in range(4)])
        src['key']=key.iloc[:nt].to_numpy()
        arr=np.full((len(x),13),np.nan,dtype=np.float32)
        def encode(ref,query):
            g=src.iloc[ref].groupby('key').agg({**{'y'+str(j):['median','mean','std'] for j in range(4)}})
            g.columns=['_'.join(c) for c in g.columns]
            g['n']=src.iloc[ref].groupby('key').size()
            return g.reindex(key.iloc[query]).to_numpy(dtype=np.float32)
        for a,b in KFold(n_splits=5,shuffle=True,random_state=18).split(it):
            arr[it[b]]=encode(it[a],it[b])
        query=np.setdiff1d(np.arange(len(x)),it)
        arr[query]=encode(it,query)
        vals.append(arr)
        print('AGGREGATE',cols,flush=True)
    result=np.hstack(vals)
    np.save(dest,result)
    return result

if __name__=='__main__':make_aggregates()
