#!/usr/bin/env python3
"""E44: test latent communication on independently randomized balanced worlds."""
from __future__ import annotations
import argparse, json, random, time
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

FACT_COUNT, VALUE_COUNT, LATENT_DIM, WIDTH = 2048, 64, 16, 64
TRAIN_COUNT = FACT_COUNT * 3 // 4

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def world(dataset_seed):
    rng=random.Random(dataset_seed)
    tr=[v for v in range(VALUE_COUNT) for _ in range(24)]
    te=[v for v in range(VALUE_COUNT) for _ in range(8)]
    rng.shuffle(tr); rng.shuffle(te)
    vals=tr+te
    return vals,[(i,vals[i]) for i in range(TRAIN_COUNT)],[(i,vals[i]) for i in range(TRAIN_COUNT,FACT_COUNT)]

class SenderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fact=nn.Embedding(FACT_COUNT,WIDTH); self.value=nn.Embedding(VALUE_COUNT,WIDTH)
        self.net=nn.Sequential(nn.Linear(WIDTH*2,WIDTH),nn.GELU(),nn.Linear(WIDTH,WIDTH),nn.GELU())
    def forward(self,f,v): return self.net(torch.cat((self.fact(f),self.value(v)),dim=-1))
class ReceiverBase(nn.Module):
    def __init__(self): super().__init__(); self.head=nn.Linear(WIDTH,VALUE_COUNT)
    def forward(self,x): return self.head(x)
class SenderAdapter(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,LATENT_DIM))
    def forward(self,x): return self.net(x)
class ReceiverAdapter(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(LATENT_DIM,WIDTH),nn.GELU(),nn.Linear(WIDTH,WIDTH))
    def forward(self,x): return self.net(x)

def quantize(z,ste=False):
    scale=z.detach().abs().amax(dim=-1,keepdim=True).clamp_min(1e-8)/127
    q=torch.round(z/scale).clamp(-127,127).to(torch.int8)
    decoded=q.float()*scale
    if ste: decoded=z+(decoded-z).detach()
    return (q,scale.squeeze(-1)),decoded
def dequantize(packet): return packet[0].float()*packet[1].unsqueeze(-1)
def norm_grad(m):
    a=[p.grad.detach().norm()**2 for p in m.parameters() if p.grad is not None]
    return float(torch.stack(a).sum().sqrt()) if a else 0.0
def delta(m,initial):
    a=[(p.detach()-x).norm()**2 for p,x in zip(m.parameters(),initial)]
    return float(torch.stack(a).sum().sqrt()) if a else 0.0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-seed",type=int,default=101); p.add_argument("--model-seed",type=int,default=43)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--out",type=Path,default=None)
    a=p.parse_args()
    if a.epochs<1: raise SystemExit("--epochs must be positive")
    seed_all(a.model_seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,train,test=world(a.dataset_seed)
    sender,receiver=SenderBase().to(device),ReceiverBase().to(device)
    for m in (sender,receiver):
        for q in m.parameters(): q.requires_grad_(False)
    sa,ra=SenderAdapter().to(device),ReceiverAdapter().to(device)
    si=[x.detach().clone() for x in sa.parameters()]; ri=[x.detach().clone() for x in ra.parameters()]
    opt=torch.optim.AdamW(list(sa.parameters())+list(ra.parameters()),lr=2e-3)
    order=random.Random(a.model_seed+1); gs=gr=None; t0=time.perf_counter()
    for _ in range(a.epochs):
        order.shuffle(train)
        for j in range(0,len(train),256):
            rows=train[j:j+256]
            f=torch.tensor([x[0] for x in rows],device=device); y=torch.tensor([x[1] for x in rows],device=device)
            with torch.no_grad(): h=sender(f,y)
            _,z=quantize(sa(h),ste=True); loss=F.cross_entropy(receiver(ra(z)),y)
            opt.zero_grad(set_to_none=True); loss.backward()
            if gs is None: gs,gr=norm_grad(sa),norm_grad(ra)
            opt.step()
    train_seconds=time.perf_counter()-t0
    @torch.no_grad()
    def evaluate(kind):
        good=n=0; t=time.perf_counter()
        for j in range(0,len(test),512):
            rows=test[j:j+512]; f=torch.tensor([x[0] for x in rows],device=device); y=torch.tensor([x[1] for x in rows],device=device)
            if kind=="no_message": msg=torch.zeros((len(rows),WIDTH),device=device)
            else:
                z=sa(sender(f,y))
                if kind=="random": z=torch.randn_like(z)
                elif kind=="zero": z=torch.zeros_like(z)
                packet,_=quantize(z)
                if kind=="shuffled":
                    ix=torch.roll(torch.arange(len(rows),device=device),1); packet=(packet[0][ix],packet[1][ix])
                msg=ra(dequantize(packet))
            good+=int((receiver(msg).argmax(-1)==y).sum()); n+=y.numel()
        return {"accuracy":good/max(n,1),"seconds":time.perf_counter()-t,"examples":n}
    counts=Counter(y for _,y in test); majority=max(counts.values())/len(test)
    packet,_=quantize(torch.zeros((1,LATENT_DIM),device=device))
    wire=packet[0].numel()*packet[0].element_size()+packet[1].numel()*packet[1].element_size()
    trainable=sum(x.numel() for m in (sa,ra) for x in m.parameters())
    frozen=sum(x.numel() for m in (sender,receiver) for x in m.parameters())
    result={"experiment":"E44_randomized_balanced_world","dataset_seed":a.dataset_seed,"model_seed":a.model_seed,"device":str(device),
      "status":"synthetic randomized-world robustness test; not a pretrained-LM test and not evidence of compression advantage",
      "dataset":{"facts":FACT_COUNT,"value_classes":VALUE_COUNT,"train_facts":len(train),"heldout_facts":len(test),
       "assignment":"independently randomized balanced mapping per dataset seed","train_examples_per_class":24,"test_examples_per_class":8,
       "split":"disjoint fact identities; class balance enforced separately in train and held-out partitions"},
      "results":{"learned_quantized_latent":evaluate("learned"),"shuffled_quantized_latent":evaluate("shuffled"),
       "random_quantized_latent":evaluate("random"),"zero_quantized_latent":evaluate("zero"),
       "no_message":evaluate("no_message"),"test_majority_class":{"accuracy":majority}},
      "learning_diagnostics":{"sender_adapter_first_step_grad_l2":gs,"receiver_adapter_first_step_grad_l2":gr,
       "sender_adapter_parameter_delta_l2":delta(sa,si),"receiver_adapter_parameter_delta_l2":delta(ra,ri)},
      "accounting":{"latent_packet_bytes_per_message":wire,"latent_packet_bits_per_message":wire*8,
       "trainable_adapter_parameters":trainable,"frozen_base_parameters":frozen,"total_parameters":trainable+frozen,
       "adapter_checkpoint_bytes_fp32":trainable*4,"frozen_base_checkpoint_bytes_fp32":frozen*4,"training_seconds":train_seconds,
       "quantization":"int8 forward packet with per-example float32 scale; STE identity-gradient approximation during training"},
      "interpretation_warning":"Tests randomized balanced fact/class assignments only. Not unseen classes, pretrained LMs, general semantic communication, or compression advantage."}
    out=a.out or Path(f"runs/E44/e44_dataset_{a.dataset_seed}_model_{a.model_seed}.json")
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
