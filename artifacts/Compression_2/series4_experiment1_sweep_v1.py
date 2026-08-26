#!/usr/bin/env python3
# S4E1 capacity/shape sweep, based on series4_experiment1_v1.py.
# Usage: python3 series4_experiment1_sweep_v1.py test FILE
import argparse,csv,hashlib,json,os,struct,time
from bisect import bisect_right
from itertools import accumulate
import torch
import torch.nn as nn

torch.manual_seed(42)
torch.use_deterministic_algorithms(True,warn_only=True)
device=torch.device("cpu")
SCALE=1<<16
VOCAB=256
PAD=256
LR=.04

class NeuralCompressor(nn.Module):
    def __init__(self,embed_dim=32,hidden_dim=64,context_size=16,num_states=4):
        super().__init__()
        self.states=nn.ModuleList([nn.Sequential(
            nn.Embedding(VOCAB+1,embed_dim,padding_idx=PAD),
            nn.Flatten(1),
            nn.Linear(context_size*embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,VOCAB)) for _ in range(num_states)])
    def forward_all(self,x):
        return torch.stack([s(x) for s in self.states])
    def forward(self,x): return self.forward_all(x).mean(0)

def make_model(e,h,c,s):
    torch.manual_seed(42)
    m=NeuralCompressor(e,h,c,s).to(device)
    return m,torch.optim.SGD(m.parameters(),lr=LR),nn.CrossEntropyLoss()

def freqs(logits):
    p=torch.softmax(torch.clamp(logits,-50,50),-1)
    return ((p*(SCALE-256)).int()+1).tolist()

class Enc:
    def __init__(self): self.low=0;self.range=0xffffffff;self.out=bytearray()
    def put(self,s,f,c):
        self.range//=c[-1];self.low+=c[s]*self.range;self.range*=f[s]
        while ((self.low^(self.low+self.range))<0x1000000 or self.range<0x10000):
            if (self.low^(self.low+self.range))<0x1000000:
                self.out.append(self.low>>24);self.low=(self.low<<8)&0xffffffff;self.range=(self.range<<8)&0xffffffff
            else:
                self.range=((-self.low&0xffff)<<8);self.out.append(self.low>>24);self.low=(self.low<<8)&0xffffffff
    def finish(self):
        for _ in range(4): self.out.append(self.low>>24);self.low=(self.low<<8)&0xffffffff
        return bytes(self.out)

class Dec:
    def __init__(self,b):
        self.b=b;self.p=0;self.low=0;self.range=0xffffffff;self.code=0
        for _ in range(4):
            x=b[self.p] if self.p<len(b) else 0;self.code=(self.code<<8)|x;self.p+=1
    def get(self,f,c):
        unit=self.range//c[-1]
        count=(self.code-self.low)//unit
        s=bisect_right(c,count)-1
        if not 0<=s<len(f): raise RuntimeError(f"invalid symbol {s}")
        self.range=unit;self.low+=c[s]*self.range;self.range*=f[s]
        while ((self.low^(self.low+self.range))<0x1000000 or self.range<0x10000):
            if (self.low^(self.low+self.range))<0x1000000:
                self.low=(self.low<<8)&0xffffffff;self.range=(self.range<<8)&0xffffffff
            else:
                self.range=((-self.low&0xffff)<<8);self.low=(self.low<<8)&0xffffffff
            x=self.b[self.p] if self.p<len(self.b) else 0;self.code=((self.code<<8)|x)&0xffffffff;self.p+=1
        return s

# name, embed, hidden, context, states
def configs():
    return [
      ("baseline",32,64,16,4),
      ("all_1",1,1,1,1),("all_2",2,2,2,2),("all_4",4,4,4,4),
      ("all_8",8,8,8,8),("all_16",16,16,16,16),("all_32",32,32,32,32),
      ("tiny_hidden64",3,64,3,3),("tiny_hidden128",3,128,3,3),
      ("tiny_hidden256",3,256,3,3),("tiny_hidden314",3,314,3,3),
      ("tiny_hidden512",3,512,3,3),
      ("hidden32",32,32,16,4),("hidden128",32,128,16,4),
      ("hidden256",32,256,16,4),("hidden512",32,512,16,4),
      ("embed8_hidden512",8,512,16,4),("embed16_hidden512",16,512,16,4),
      ("embed32_hidden512",32,512,16,4),("embed64_hidden512",64,512,16,4),
      ("context4",32,64,4,4),("context8",32,64,8,4),("context32",32,64,32,4),
      ("states1",32,64,16,1),("states2",32,64,16,2),("states3",32,64,16,3),
      ("states8",32,64,16,8),
      ("odd_3_314_3_3",3,314,3,3),("odd_5_333_5_5",5,333,5,5),
      ("odd_7_127_7_3",7,127,7,3),("odd_11_257_11_3",11,257,11,3),
      ("odd_13_385_5_5",13,385,5,5),("odd_24_384_12_6",24,384,12,6),
      ("odd_48_300_10_3",48,300,10,3),("odd_64_511_17_3",64,511,17,3)]

def run(path,name,e,h,c,s,keep):
    data=open(path,"rb").read(); n=len(data); sha=hashlib.sha256(data).hexdigest()
    t=time.time();m,opt,crit=make_model(e,h,c,s);enc=Enc();ctx=[PAD]*c
    for target in data:
        x=torch.tensor([ctx],dtype=torch.long);log=m(x).squeeze(0);f=freqs(log.detach());cu=list(accumulate(f,initial=0));enc.put(target,f,cu)
        opt.zero_grad();crit(log.unsqueeze(0),torch.tensor([target])).backward();opt.step();ctx=ctx[1:]+[target]
    payload=enc.finish();enc_t=time.time()-t; blob=struct.pack(">I B B",n,1,s)+payload
    t=time.time();m,opt,crit=make_model(e,h,c,s);dec=Dec(payload);ctx=[PAD]*c;out=bytearray();err=""
    try:
        for _ in range(n):
            x=torch.tensor([ctx],dtype=torch.long);log=m(x).squeeze(0);f=freqs(log.detach());cu=list(accumulate(f,initial=0));z=dec.get(f,cu);out.append(z)
            opt.zero_grad();crit(log.unsqueeze(0),torch.tensor([z])).backward();opt.step();ctx=ctx[1:]+[z]
    except Exception as ex: err=f"{type(ex).__name__}: {ex}"
    dec_t=time.time()-t; ok=not err and hashlib.sha256(out).hexdigest()==sha
    red=(1-len(blob)/n)*100
    r=dict(name=name,embed_dim=e,hidden_dim=h,context_size=c,num_states=s,raw_bytes=n,compressed_bytes=len(blob),reduction_percent=red,encode_seconds=enc_t,decode_seconds=dec_t,roundtrip_seconds=enc_t+dec_t,exact_match=ok,error=err)
    print(f"{name:22} e={e:3} h={h:3} c={c:2} s={s:2} -> {len(blob):6} B {red:6.2f}% rt={enc_t+dec_t:7.2f}s {'PASS' if ok else 'FAIL'}")
    if keep:
        open(path+".s4sweep."+name+".bin","wb").write(blob);open(path+".s4sweep."+name+".recon","wb").write(out)
    return r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=["test"]);ap.add_argument("file")
    ap.add_argument("--limit",type=int);ap.add_argument("--keep-files",action="store_true")
    ap.add_argument("--configs",help="e,h,c,s;e,h,c,s;...")
    a=ap.parse_args()
    cs=configs()
    if a.configs:
        cs=[(f"custom{i}",*map(int,x.split(","))) for i,x in enumerate(a.configs.split(";"),1)]
    if a.limit: cs=cs[:a.limit]
    print(f"S4E1 sweep: {len(cs)} configs; vocab=256 lr={LR}")
    results=[]
    for q in cs:
        try: results.append(run(a.file,*q,a.keep_files))
        except Exception as ex: results.append(dict(name=q[0],embed_dim=q[1],hidden_dim=q[2],context_size=q[3],num_states=q[4],error=f"{type(ex).__name__}: {ex}",exact_match=False));print(q[0],"FAILED",ex)
    good=sorted([r for r in results if r.get("exact_match")],key=lambda x:x["compressed_bytes"])
    print("\nBEST EXACT:")
    for i,r in enumerate(good,1): print(f"{i:2}. {r['name']:22} {r['compressed_bytes']:6} B {r['reduction_percent']:6.2f}%")
    base=a.file+".s4e1_sweep"
    with open(base+".csv","w",newline="") as f:
        fields=sorted({k for r in results for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    json.dump({"experiment":"S4E1 capacity_shape_v1","vocab_size":VOCAB,"lr":LR,"seed":42,"results":results},open(base+".json","w"),indent=2)
    print("CSV:",base+".csv");print("JSON:",base+".json")
if __name__=="__main__": main()
