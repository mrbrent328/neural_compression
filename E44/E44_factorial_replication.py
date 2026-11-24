#!/usr/bin/env python3
"""Cross randomized dataset assignments with model seeds for E44."""
from __future__ import annotations
import argparse,json,statistics,subprocess,sys
from pathlib import Path
METRICS=("learned_quantized_latent","shuffled_quantized_latent","random_quantized_latent","zero_quantized_latent","no_message")
def stats(xs): return {"n":len(xs),"mean":statistics.mean(xs),"sample_stddev":statistics.stdev(xs) if len(xs)>1 else 0.0,"median":statistics.median(xs),"min":min(xs),"max":max(xs)}
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset-seeds",nargs="+",type=int,default=[101,102,103,104,105])
 p.add_argument("--model-seeds",nargs="+",type=int,default=[43,44,45,46,47]); p.add_argument("--epochs",type=int,default=60)
 p.add_argument("--script",type=Path,default=Path(__file__).with_name("E44_randomized_world.py"))
 p.add_argument("--run-dir",type=Path,default=Path("runs/E44/factorial")); p.add_argument("--out",type=Path,default=Path("runs/E44/e44_factorial_summary.json"))
 a=p.parse_args()
 if not a.dataset_seeds or not a.model_seeds or a.epochs<1: raise SystemExit("Provide seeds and positive epochs")
 a.run_dir.mkdir(parents=True,exist_ok=True); runs=[]
 for ds in a.dataset_seeds:
  for ms in a.model_seeds:
   raw=a.run_dir/f"e44_dataset_{ds}_model_{ms}.json"
   cmd=[sys.executable,str(a.script),"--dataset-seed",str(ds),"--model-seed",str(ms),"--epochs",str(a.epochs),"--out",str(raw)]
   print(f"Running dataset_seed={ds}, model_seed={ms}",flush=True)
   done=subprocess.run(cmd,text=True,capture_output=True,check=False)
   if done.returncode: raise RuntimeError(f"Run failed ({ds},{ms})\n{done.stdout}\n{done.stderr}")
   d=json.loads(raw.read_text())
   if d.get("experiment")!="E44_randomized_balanced_world" or d.get("dataset_seed")!=ds or d.get("model_seed")!=ms: raise RuntimeError(f"Unexpected output identity: {raw}")
   runs.append({"dataset_seed":ds,"model_seed":ms,"raw_result":str(raw),"results":{k:d["results"][k] for k in METRICS}})
   print(f"  learned={d['results']['learned_quantized_latent']['accuracy']:.4%}",flush=True)
 agg={k:stats([r["results"][k]["accuracy"] for r in runs]) for k in METRICS}; paired={}
 for c in METRICS[1:]:
  diffs=[r["results"]["learned_quantized_latent"]["accuracy"]-r["results"][c]["accuracy"] for r in runs]
  paired["learned_minus_"+c]={**stats(diffs),"runs_learned_higher":sum(x>0 for x in diffs),"runs_tied":sum(x==0 for x in diffs),"runs_learned_lower":sum(x<0 for x in diffs)}
 byds={}
 for ds in a.dataset_seeds:
  rr=[r for r in runs if r["dataset_seed"]==ds]
  byds[str(ds)]={"n_model_seeds":len(rr),"learned_accuracy":stats([r["results"]["learned_quantized_latent"]["accuracy"] for r in rr])}
 out={"experiment":"E44_factorial_randomized_world_replication","status":"synthetic randomized-world robustness only; not pretrained-LM or compression evidence",
  "protocol":{"dataset_seeds":a.dataset_seeds,"model_seeds":a.model_seeds,"epochs_per_run":a.epochs,"runs":len(runs),"crossed_design":True,
   "independent_process_per_combination":True,"dataset_assignments":"new balanced mapping per dataset seed; exactly 24 train and 8 held-out examples per class",
   "model_randomness":"independent model initialization and training-order seed"},
  "aggregate_accuracy":agg,"paired_accuracy_differences":paired,"by_dataset_seed":byds,"runs":runs,
  "interpretation_warning":"This is robustness across controlled synthetic assignments and initializations, not 25 independent natural datasets. Descriptive summaries only; do not infer statistical significance."}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+"\n")
 print(json.dumps({"summary_path":str(a.out),"runs":len(runs),"aggregate_accuracy":agg,"paired_accuracy_differences":paired},indent=2))
if __name__=="__main__": main()
