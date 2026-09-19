import os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.runtime'))
os.environ['HF_HOME']=str(ROOT/'.cache'/'huggingface')
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModel

def run():
    torch.set_num_threads(4)
    print('TORCH',torch.__version__,torch.cuda.is_available(),flush=True)
    name='intfloat/multilingual-e5-small'
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModel.from_pretrained(name).to('cuda').half().eval()
    print('MODEL_READY',flush=True)
    df=pd.concat([pd.read_parquet(ROOT/'train.parquet'),pd.read_parquet(ROOT/'test.parquet')],ignore_index=True)
    titles=df.title.fillna('').str.slice(0,250)
    descriptions=df.description.fillna('').str.replace(r'\s+',' ',regex=True).str.slice(0,900)
    texts=('passage: '+df.subcategory_name+'. '+df.microcat_name+'. '+titles+'. '+titles+'. '+descriptions).tolist()
    dest=ROOT/'artifacts'/'embeddings.npy'
    output=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(df),384))
    batch_size=192; start=time.time()
    order=np.argsort([len(s) for s in texts])
    with torch.inference_mode():
        for a in range(0,len(texts),batch_size):
            ids=order[a:a+batch_size]
            tokens=tokenizer([texts[i] for i in ids],max_length=160,padding=True,truncation=True,return_tensors='pt').to('cuda')
            h=model(**tokens).last_hidden_state.float()
            mask=tokens['attention_mask'].unsqueeze(-1)
            emb=(h*mask).sum(1)/mask.sum(1)
            emb=torch.nn.functional.normalize(emb,p=2,dim=1)
            output[ids]=emb.cpu().numpy()
            if a%(batch_size*50)==0:print('EMBED',a,'/',len(texts),'seconds',round(time.time()-start),flush=True)
    output.flush(); print('DONE',time.time()-start,flush=True)

if __name__=='__main__':run()
