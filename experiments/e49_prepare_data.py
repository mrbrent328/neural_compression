#!/usr/bin/env python3
"""Build deterministic whole-source-file-isolated uint32 token splits for E49."""
import argparse,hashlib,json,random
from pathlib import Path
EXTS={".txt",".md",".rst",".html",".xml",".json",".csv",".py",".log"}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--data",type=Path,required=True); p.add_argument("--tokenizer",default="Qwen/Qwen2.5-0.5B")
 p.add_argument("--output-dir",type=Path,default=Path("data/e49_tokens")); p.add_argument("--seed",type=int,default=4901)
 p.add_argument("--min-file-bytes",type=int,default=256); a=p.parse_args()
 try:
  import numpy as np
  from transformers import AutoTokenizer
 except ImportError as e: raise SystemExit("Install: pip install numpy transformers tokenizers") from e
 paths=[a.data] if a.data.is_file() else sorted(x for x in a.data.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)
 rows=[]; seen=set()
 for f in paths:
  raw=f.read_bytes(); h=hashlib.sha256(raw).hexdigest()
  if len(raw)>=a.min_file_bytes and h not in seen: rows.append((f,h)); seen.add(h)
 if len(rows)<3: raise SystemExit("Need at least 3 distinct source files for file-isolated train/validation/test.")
 random.Random(a.seed).shuffle(rows); n=len(rows); nv=max(1,round(n*.01)); nt=max(1,round(n*.01))
 if nv+nt>=n: nv=nt=1
 splits={"train":rows[:n-nv-nt],"validation":rows[n-nv-nt:n-nt],"test":rows[n-nt:]}
 tok=AutoTokenizer.from_pretrained(a.tokenizer,use_fast=True); a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest={"experiment":"E49","tokenizer":a.tokenizer,"vocab_size":len(tok),"seed":a.seed,"split_method":"whole source files; exact duplicate files removed","splits":{}}
 for name,items in splits.items():
  path=a.output_dir/f"{name}.bin"; count=0
  with path.open("wb") as out:
   for f,h in items:
    ids=np.asarray(tok.encode(f.read_text(encoding="utf-8",errors="replace"),add_special_tokens=False),dtype=np.uint32)
    ids.tofile(out); count+=len(ids)
  manifest["splits"][name]={"tokens":count,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"sources":[{"path":str(f),"sha256":h} for f,h in items]}
  if count<1_000_000: print(f"WARNING: {name} has only {count:,} tokens; repeated sampling risks memorization.")
 (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
