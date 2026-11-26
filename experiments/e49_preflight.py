#!/usr/bin/env python3
"""E49 model-family parameter and rough FSDP memory preflight."""
import argparse,json
from e49_model import E49Config,E49LM
def main():
 p=argparse.ArgumentParser()
 for name,default in [("vocab-size",32000),("layers",32),("d-model",4096),("heads",32),("kv-heads",32),("d-ff",11008),("context",4096),("rank",512),("latent-dim",16),("gpus",8)]:
  p.add_argument("--"+name,type=int,default=default)
 a=p.parse_args(); rows=[]
 for mode in ("dense","lowrank","latent"):
  c=E49Config(vocab_size=a.vocab_size,n_layers=a.layers,d_model=a.d_model,n_heads=a.heads,n_kv_heads=a.kv_heads,d_ff=a.d_ff,context=a.context,mode=mode,rank=a.rank,latent_dim=a.latent_dim)
  with __import__("torch").device("meta"): m=E49LM(c)
  n=m.parameter_breakdown()["total"]
  rows.append({"mode":mode,"parameters":m.parameter_breakdown(),"fp32_checkpoint_weights_gib":n*4/1024**3,
   "rough_sharded_fp32_parameters_gradients_adam_moments_gib_per_gpu":n*16/a.gpus/1024**3})
 print(json.dumps({"config":vars(a),"estimates":rows,"warning":"Assumes FP32 sharded parameters + FP32 gradients + two FP32 Adam moments (16 bytes/parameter). Excludes activations, FSDP all-gathers, temporary generated factors, CUDA context and fragmentation."},indent=2))
if __name__=="__main__": main()
