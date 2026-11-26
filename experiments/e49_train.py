#!/usr/bin/env python3
"""E49 BF16/FSDP training runner. Launch one process per GPU with torchrun."""
import argparse,hashlib,json,math,os,random,time
from functools import partial
from pathlib import Path
import numpy as np
import torch,torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision,ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict,StateDictOptions
from e49_model import E49Config,E49LM,DecoderBlock,LowRankLinear,LatentLinear,RMSNorm

def args_parse():
 p=argparse.ArgumentParser()
 p.add_argument("--mode",choices=["dense","lowrank","latent"],required=True)
 p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
 p.add_argument("--vocab-size",type=int,required=True)
 for n,d in [("layers",32),("d-model",4096),("heads",32),("kv-heads",32),("d-ff",11008),("context",4096),("rank",512),("latent-dim",16),("constructor-hidden",256),("constructor-type-dim",32),("constructor-fourier-bands",8)]:
  p.add_argument("--"+n,type=int,default=d)
 p.add_argument("--tokens",type=int,default=20_000_000_000); p.add_argument("--micro-batch",type=int,default=1); p.add_argument("--grad-accum",type=int,default=8)
 p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--min-lr-ratio",type=float,default=.1); p.add_argument("--weight-decay",type=float,default=.1)
 p.add_argument("--warmup-tokens",type=int,default=200_000_000); p.add_argument("--checkpoint-tokens",type=int,default=500_000_000)
 p.add_argument("--eval-tokens",type=int,default=2_000_000); p.add_argument("--eval-every-steps",type=int,default=500)
 p.add_argument("--log-every",type=int,default=20); p.add_argument("--seed",type=int,default=4901)
 p.add_argument("--resume",action="store_true"); p.add_argument("--activation-checkpointing",action="store_true"); p.add_argument("--dry-run",action="store_true")
 return p.parse_args()

def load_tokens(path):
 if not path.exists(): raise SystemExit(f"Missing token file: {path}")
 a=np.memmap(path,dtype=np.uint32,mode="r")
 if a.size<4096: raise SystemExit(f"Too few tokens in {path}: {a.size}")
 return a

def batch(data,n,ctx,device,rng):
 starts=torch.randint(0,len(data)-ctx-1,(n,),generator=rng).numpy()
 arr=np.stack([data[i:i+ctx+1].astype(np.int64) for i in starts])
 t=torch.from_numpy(arr).to(device=device,dtype=torch.long,non_blocking=True)
 return t[:,:-1],t[:,1:]

def evaluate(model,data,args,device,rng,steps):
 model.eval(); total=torch.zeros(2,device=device,dtype=torch.float64)
 with torch.no_grad():
  for _ in range(steps):
   x,y=batch(data,args.micro_batch,args.context,device,rng)
   with torch.autocast("cuda",dtype=torch.bfloat16): _,loss=model(x,y)
   total[0]+=loss.detach().double(); total[1]+=1
 if dist.is_initialized(): dist.all_reduce(total)
 model.train(); return (total[0]/total[1].clamp_min(1)).item()

def save_ckpt(model,opt,path,meta):
 path.mkdir(parents=True,exist_ok=True)
 ms,osd=get_state_dict(model,opt,options=StateDictOptions(full_state_dict=False,cpu_offload=True))
 dcp.save({"model":ms,"optimizer":osd},checkpoint_id=path/"shards")
 if not dist.is_initialized() or dist.get_rank()==0: (path/"trainer_state.json").write_text(json.dumps(meta,indent=2)+"\n")
 if dist.is_initialized(): dist.barrier()

def load_ckpt(model,opt,path):
 ms,osd=get_state_dict(model,opt,options=StateDictOptions(full_state_dict=False,cpu_offload=True))
 state={"model":ms,"optimizer":osd}
 dcp.load(state,checkpoint_id=path/"shards")
 set_state_dict(model,opt,model_state_dict=state["model"],optim_state_dict=state["optimizer"],
  options=StateDictOptions(full_state_dict=False,cpu_offload=True))
 if dist.is_initialized(): dist.barrier()
 return json.loads((path/"trainer_state.json").read_text())

def main():
 a=args_parse()
 cfg=E49Config(vocab_size=a.vocab_size,n_layers=a.layers,d_model=a.d_model,n_heads=a.heads,n_kv_heads=a.kv_heads,d_ff=a.d_ff,context=a.context,mode=a.mode,rank=a.rank,latent_dim=a.latent_dim,constructor_hidden=a.constructor_hidden,constructor_type_dim=a.constructor_type_dim,constructor_fourier_bands=a.constructor_fourier_bands)
 cfg.validate()
 if min(a.tokens,a.micro_batch,a.grad_accum,a.eval_every_steps,a.checkpoint_tokens)<=0: raise SystemExit("tokens, batch sizes, and intervals must be positive")
 world=int(os.environ.get("WORLD_SIZE","1"))
 tokstep=a.micro_batch*a.context*a.grad_accum*world; steps=math.ceil(a.tokens/tokstep)
 if a.dry_run:
  with torch.device("meta"): m=E49LM(cfg)
  print(json.dumps({"config":cfg.to_dict(),"parameters":m.parameter_breakdown(),"tokens_per_optimizer_step":tokstep,"planned_steps":steps,"dry_run":True},indent=2)); return
 if not torch.cuda.is_available(): raise SystemExit("Training requires CUDA/ROCm-visible GPUs. Use --dry-run for CPU preflight.")
 if not torch.cuda.is_bf16_supported(): raise SystemExit("Selected accelerator does not report BF16 support; E49 currently requires BF16.")
 local=int(os.environ.get("LOCAL_RANK","0")); torch.cuda.set_device(local)
 dist.init_process_group("nccl")
 device=torch.device("cuda",local)
 rank=dist.get_rank()
 seed=a.seed; random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 manifest_path=a.data_dir/"manifest.json"
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  if int(manifest.get("vocab_size",-1)) != a.vocab_size:
   raise SystemExit(f"--vocab-size {a.vocab_size} does not match tokenizer manifest {manifest.get('vocab_size')}")
 tr=load_tokens(a.data_dir/"train.bin"); va=load_tokens(a.data_dir/"validation.bin")
 with torch.device("meta"):
  model=E49LM(cfg)
 def materialize_init(module):
  own_params=list(module.parameters(recurse=False))
  own_buffers=list(module.buffers(recurse=False))
  if not any(x.is_meta for x in own_params+own_buffers): return
  module.to_empty(device=device,recurse=False)
  if isinstance(module,nn.Linear):
   std=.002 if module.out_features==cfg.rank and module.in_features==cfg.constructor_hidden else .02
   nn.init.normal_(module.weight,mean=0,std=std)
   if module.bias is not None: nn.init.zeros_(module.bias)
  elif isinstance(module,nn.Embedding): nn.init.normal_(module.weight,mean=0,std=.02)
  elif isinstance(module,RMSNorm): nn.init.ones_(module.weight)
  elif isinstance(module,LowRankLinear):
   nn.init.kaiming_uniform_(module.right,a=math.sqrt(5)); nn.init.normal_(module.left,std=.02/math.sqrt(module.right.shape[0]))
  elif isinstance(module,LatentLinear):
   nn.init.normal_(module.z,std=.02); nn.init.zeros_(module.log_gain)
 if a.activation_checkpointing:
  from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper,CheckpointImpl,apply_activation_checkpointing
  wrap=partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
  apply_activation_checkpointing(model,checkpoint_wrapper_fn=wrap,check_fn=lambda m:isinstance(m,DecoderBlock))
 from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
 auto=partial(transformer_auto_wrap_policy,transformer_layer_cls={DecoderBlock,CheckpointWrapper})
 mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
 model=FSDP(model,auto_wrap_policy=auto,sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=mp,device_id=torch.device("cuda",local),use_orig_params=True,limit_all_gathers=True,param_init_fn=materialize_init)
 decay=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
 no_decay=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
 opt=torch.optim.AdamW([{"params":decay,"weight_decay":a.weight_decay},{"params":no_decay,"weight_decay":0.0}],
  lr=a.lr,betas=(.9,.95),fused=True)
 start,tokens=0,0
 a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.resume:
  meta=load_ckpt(model,opt,a.output_dir/"latest"); start=int(meta["step"]); tokens=int(meta["tokens_seen"])
 rng=torch.Generator(device="cpu").manual_seed(a.seed+17+rank); erng=torch.Generator(device="cpu").manual_seed(a.seed+23)
 manifest_hash=hashlib.sha256((a.data_dir/"manifest.json").read_bytes()).hexdigest() if (a.data_dir/"manifest.json").exists() else None
 last_ckpt=tokens; initial_tokens=tokens; t0=time.time(); model.train()
 metrics_path=a.output_dir/"metrics.jsonl"
 for step in range(start,steps):
  if tokens<a.warmup_tokens: lr=a.lr*(tokens+1)/max(1,a.warmup_tokens)
  else:
   d=(tokens-a.warmup_tokens)/max(1,a.tokens-a.warmup_tokens)
   lr=a.lr*(a.min_lr_ratio+(1-a.min_lr_ratio)*.5*(1+math.cos(math.pi*min(1,d))))
  for g in opt.param_groups:g["lr"]=lr
  opt.zero_grad(set_to_none=True); loss_sum=0.
  for _ in range(a.grad_accum):
   x,y=batch(tr,a.micro_batch,a.context,torch.device("cuda",local),rng)
   with torch.autocast("cuda",dtype=torch.bfloat16): _,loss=model(x,y); loss=loss/a.grad_accum
   loss.backward(); loss_sum+=loss.detach().float().item()
  model.clip_grad_norm_(1.0); opt.step(); tokens+=tokstep
  if (step+1)%a.log_every==0 and rank==0:
   print(f"step={step+1:,}/{steps:,} tokens={tokens:,} loss={loss_sum:.4f} lr={lr:.3g} tok/s={tokens/max(1,time.time()-t0):,.0f}",flush=True)
  if (step+1)%a.eval_every_steps==0:
   n=max(1,a.eval_tokens//(a.micro_batch*a.context*world))
   val=evaluate(model,va,a,torch.device("cuda",local),erng,n)
   if rank==0:
    row={"type":"validation","step":step+1,"tokens_seen":tokens,"val_loss":val,"perplexity":math.exp(min(20,val))}
    with metrics_path.open("a") as f: f.write(json.dumps(row)+"\\n")
    print(f"EVAL step={step+1:,} tokens={tokens:,} val_loss={val:.6f} ppl={math.exp(min(20,val)):.4f}",flush=True)
  if tokens-last_ckpt>=a.checkpoint_tokens or step+1==steps:
   meta={"experiment":"E49","mode":a.mode,"step":step+1,"tokens_seen":tokens,"world_size":world,"config":cfg.to_dict(),"elapsed_seconds":time.time()-t0,"data_manifest_sha256":manifest_hash,"peak_memory_allocated":torch.cuda.max_memory_allocated(device),"peak_memory_reserved":torch.cuda.max_memory_reserved(device)}
   save_ckpt(model,opt,a.output_dir/"latest",meta); last_ckpt=tokens
   if rank==0: print(f"checkpoint saved at {tokens:,} tokens",flush=True)
 if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()
if __name__=="__main__":main()
