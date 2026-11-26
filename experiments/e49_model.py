#!/usr/bin/env python3
"""E49 scalable decoder LM: dense, direct low-rank, shared-constructor MLPs."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

@dataclass
class E49Config:
    vocab_size: int = 32000
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    d_ff: int = 11008
    context: int = 4096
    rope_theta: float = 10000.0
    mode: str = "latent"
    rank: int = 512
    latent_dim: int = 16
    constructor_hidden: int = 256
    constructor_type_dim: int = 32
    constructor_fourier_bands: int = 8
    tie_embeddings: bool = False

    def validate(self):
        if self.mode not in {"dense", "lowrank", "latent"}: raise ValueError(self.mode)
        if self.d_model % self.n_heads: raise ValueError("d_model must divide n_heads")
        if self.n_heads % self.n_kv_heads: raise ValueError("n_heads must divide n_kv_heads")
        if (self.d_model // self.n_heads) % 2: raise ValueError("RoPE requires even head_dim")
        if self.rank > min(self.d_model, self.d_ff): raise ValueError("rank too large")
        if self.context < 2 or self.n_layers < 1: raise ValueError("invalid dimensions")

    def to_dict(self): return asdict(self)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        n=x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return n.to(x.dtype)*self.weight

def apply_rope(q,k,theta):
    d=q.shape[-1]
    inv=1.0/(theta**(torch.arange(0,d,2,device=q.device,dtype=torch.float32)/d))
    pos=torch.arange(q.shape[-2],device=q.device,dtype=torch.float32)
    f=torch.outer(pos,inv); c=f.cos().to(q.dtype)[None,None]; s=f.sin().to(q.dtype)[None,None]
    def rotate(x):
        a,b=x[...,:d//2],x[...,d//2:]
        return torch.cat((a*c-b*s,b*c+a*s),dim=-1)
    return rotate(q),rotate(k)

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.h,self.kh=cfg.n_heads,cfg.n_kv_heads
        self.hd=cfg.d_model//cfg.n_heads; self.theta=cfg.rope_theta
        self.q=nn.Linear(cfg.d_model,self.h*self.hd,bias=False)
        self.k=nn.Linear(cfg.d_model,self.kh*self.hd,bias=False)
        self.v=nn.Linear(cfg.d_model,self.kh*self.hd,bias=False)
        self.o=nn.Linear(self.h*self.hd,cfg.d_model,bias=False)
    def forward(self,x):
        b,t,_=x.shape
        q=self.q(x).view(b,t,self.h,self.hd).transpose(1,2)
        k=self.k(x).view(b,t,self.kh,self.hd).transpose(1,2)
        v=self.v(x).view(b,t,self.kh,self.hd).transpose(1,2)
        q,k=apply_rope(q,k,self.theta)
        if self.kh!=self.h:
            r=self.h//self.kh; k=k.repeat_interleave(r,1); v=v.repeat_interleave(r,1)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(b,t,-1))

class LowRankLinear(nn.Module):
    def __init__(self,ni,no,r):
        super().__init__(); self.right=nn.Parameter(torch.empty(r,ni)); self.left=nn.Parameter(torch.empty(no,r))
        nn.init.kaiming_uniform_(self.right,a=math.sqrt(5)); nn.init.normal_(self.left,std=.02/math.sqrt(r))
    def forward(self,x): return F.linear(F.linear(x,self.right),self.left)

class SharedCoordinateConstructor(nn.Module):
    """One shared interpreter; parameter count is independent of matrix/layer count."""
    def __init__(self,cfg):
        super().__init__(); self.rank=cfg.rank; self.bands=cfg.constructor_fourier_bands
        self.type_embedding=nn.Embedding(3,cfg.constructor_type_dim)
        dim=cfg.latent_dim+cfg.constructor_type_dim+1+2*self.bands
        def net():
            m=nn.Sequential(nn.Linear(dim,cfg.constructor_hidden),nn.SiLU(),
                nn.Linear(cfg.constructor_hidden,cfg.constructor_hidden),nn.SiLU(),
                nn.Linear(cfg.constructor_hidden,cfg.rank))
            nn.init.normal_(m[-1].weight,std=.002); nn.init.zeros_(m[-1].bias); return m
        self.row_net,self.col_net=net(),net()
        self.register_buffer("coords_d",self._coords(cfg.d_model),persistent=False)
        self.register_buffer("coords_ff",self._coords(cfg.d_ff),persistent=False)
    def _coords(self,n):
        x=torch.linspace(-1,1,n).unsqueeze(-1); out=[x]
        for i in range(self.bands):
            f=math.pi*(2**i); out.extend((torch.sin(f*x),torch.cos(f*x)))
        return torch.cat(out,-1)
    def _inputs(self,z,kind,coords):
        c=coords.to(device=z.device,dtype=z.dtype); e=self.type_embedding.weight[kind].to(z.dtype)
        return torch.cat((z.reshape(1,-1).expand(c.shape[0],-1),
            e.reshape(1,-1).expand(c.shape[0],-1),c),-1)
    def _coords_for(self,n):
        if n == self.coords_d.shape[0]: return self.coords_d
        if n == self.coords_ff.shape[0]: return self.coords_ff
        return self._coords(n)
    def factors(self,z,kind,ni,no):
        rc=self._coords_for(no)
        cc=self._coords_for(ni)
        left=self.row_net(self._inputs(z,kind,rc))/math.sqrt(self.rank)
        right=self.col_net(self._inputs(z,kind,cc)).transpose(0,1).contiguous()
        return left,right

class LatentLinear(nn.Module):
    def __init__(self,ni,no,cfg,kind):
        super().__init__(); self.ni,self.no,self.kind=ni,no,kind
        self.z=nn.Parameter(torch.randn(cfg.latent_dim)*.02); self.log_gain=nn.Parameter(torch.zeros(()))
    def forward(self,x,constructor):
        l,r=constructor.factors(self.z,self.kind,self.ni,self.no)
        return F.linear(F.linear(x,r),l)*self.log_gain.exp()

class SwiGLU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg.mode=="dense":
            self.gate=nn.Linear(cfg.d_model,cfg.d_ff,bias=False); self.up=nn.Linear(cfg.d_model,cfg.d_ff,bias=False); self.down=nn.Linear(cfg.d_ff,cfg.d_model,bias=False)
        elif cfg.mode=="lowrank":
            self.gate=LowRankLinear(cfg.d_model,cfg.d_ff,cfg.rank); self.up=LowRankLinear(cfg.d_model,cfg.d_ff,cfg.rank); self.down=LowRankLinear(cfg.d_ff,cfg.d_model,cfg.rank)
        else:
            self.gate=LatentLinear(cfg.d_model,cfg.d_ff,cfg,0); self.up=LatentLinear(cfg.d_model,cfg.d_ff,cfg,1); self.down=LatentLinear(cfg.d_ff,cfg.d_model,cfg,2)
    def forward(self,x,constructor):
        if constructor is None: return self.down(F.silu(self.gate(x))*self.up(x))
        return self.down(F.silu(self.gate(x,constructor))*self.up(x,constructor),constructor)

class DecoderBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.attn_norm=RMSNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg)
        self.mlp_norm=RMSNorm(cfg.d_model); self.mlp=SwiGLU(cfg)
    def forward(self,x,constructor):
        x=x+self.attn(self.attn_norm(x)); return x+self.mlp(self.mlp_norm(x),constructor)

class E49LM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); cfg.validate(); self.cfg=cfg
        self.embed_tokens=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.layers=nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm=RMSNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        if cfg.tie_embeddings: self.lm_head.weight=self.embed_tokens.weight
        self.constructor=SharedCoordinateConstructor(cfg) if cfg.mode=="latent" else None
        self.apply(self._init_weights)
        if self.constructor:
            for m in (self.constructor.row_net,self.constructor.col_net):
                nn.init.normal_(m[-1].weight,std=.002); nn.init.zeros_(m[-1].bias)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,mean=0,std=.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,mean=0,std=.02)
    def forward(self,input_ids,labels=None):
        x=self.embed_tokens(input_ids)
        for layer in self.layers: x=layer(x,self.constructor)
        logits=self.lm_head(self.norm(x)); loss=None
        if labels is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1))
        return logits,loss
    def parameter_breakdown(self):
        out={"total":0,"attention":0,"mlp":0,"latents":0,"constructor":0,"embeddings_and_norms":0}
        for name,p in self.named_parameters():
            n=p.numel(); out["total"]+=n
            if "constructor." in name: out["constructor"]+=n
            elif ".mlp." in name:
                out["latents" if name.endswith(".z") or name.endswith(".log_gain") else "mlp"]+=n
            elif ".attn." in name: out["attention"]+=n
            else: out["embeddings_and_norms"]+=n
        return out
