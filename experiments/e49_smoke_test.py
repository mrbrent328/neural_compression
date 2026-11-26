#!/usr/bin/env python3
"""CPU forward/backward smoke test for all E49 MLP parameterizations."""
import torch
from e49_model import E49Config,E49LM

def main():
 torch.manual_seed(49)
 for mode in ("dense","lowrank","latent"):
  cfg=E49Config(vocab_size=128,n_layers=2,d_model=64,n_heads=4,n_kv_heads=2,
   d_ff=128,context=32,mode=mode,rank=16,latent_dim=8,
   constructor_hidden=32,constructor_type_dim=8,constructor_fourier_bands=3)
  model=E49LM(cfg)
  ids=torch.randint(0,cfg.vocab_size,(2,17))
  logits,loss=model(ids[:,:-1],ids[:,1:])
  assert logits.shape==(2,16,cfg.vocab_size), (mode,logits.shape)
  assert loss is not None and torch.isfinite(loss), (mode,loss)
  loss.backward()
  missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
  if missing: raise AssertionError(f"{mode}: parameters without gradients: {missing[:8]}")
  print(f"{mode}: OK | params={model.parameter_breakdown()['total']:,} | loss={loss.item():.4f}")
 print("E49 model smoke test passed for dense, lowrank, and latent.")

if __name__=="__main__": main()
