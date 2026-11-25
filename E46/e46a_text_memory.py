#!/usr/bin/env python3
"""E46-A: deterministic persistent text-memory benchmark. Prepare and evaluate are separate process commands."""
import argparse, json, os, time, urllib.request
from pathlib import Path

PEOPLE = [
 ("Neris","copper","oboe","North Fen"),("Talven","indigo","cello","Glass Coast"),
 ("Orika","saffron","flute","Moss Reach"),("Perrin","silver","harp","Ash Valley"),
 ("Suvia","teal","violin","Cloud March"),("Daro","crimson","clarinet","Stone Basin"),
 ("Ilyra","amber","bassoon","Reed Plain"),("Vey","violet","trumpet","Frost Garden")]
OUT=Path("runs/E46A")

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2)+"\n")

def prepare(out):
 facts=[dict(key=n,color=c,instrument=i,region=r) for n,c,i,r in PEOPLE]
 qs=[]
 for x in facts:
  n=x["key"]
  qs.extend([
   dict(id=n+"-A",kind="exact",question="What color is assigned to "+n+"?",answer=x["color"]),
   dict(id=n+"-B",kind="paraphrase",question="Which musical instrument belongs to "+n+"?",answer=x["instrument"]),
   dict(id=n+"-U",kind="unsupported",question="What pet does "+n+" own?",answer="UNKNOWN"),
   dict(id=n+"-R",kind="relational",question="Who is assigned to the "+x["region"]+" region?",answer=n)])
 shuffled=[]
 for i,x in enumerate(facts):
  y=facts[(i+1)%len(facts)]
  shuffled.append(dict(key=x["key"],color=y["color"],instrument=y["instrument"],region=y["region"]))
 save(out/"facts.json",facts); save(out/"questions.json",qs); save(out/"shuffled.json",shuffled)
 (out/"raw_memory.txt").write_text("\n".join(
  x["key"]+": color "+x["color"]+"; instrument "+x["instrument"]+"; region "+x["region"]+"."
  for x in facts)+"\n")
 save(out/"manifest.json",dict(experiment="E46-A",facts=8,questions=32,
  categories=["exact","paraphrase","relational","unsupported"],
  conditions=["no_memory","raw_text_memory","shuffled_mismatched_memory"]))
 print("Prepared 8 facts and 32 questions. Stop this process; run evaluate in a fresh process.")

def call(url,payload=None,key=None,timeout=300):
 h={"Accept":"application/json"}
 if payload is not None: h["Content-Type"]="application/json"
 if key: h["Authorization"]="Bearer "+key
 data=json.dumps(payload).encode() if payload is not None else None
 req=urllib.request.Request(url,data=data,headers=h)
 with urllib.request.urlopen(req,timeout=timeout) as r: return json.loads(r.read().decode())

def evaluate(a):
 out=a.output
 facts=json.loads((out/"facts.json").read_text())
 qs=json.loads((out/"questions.json").read_text())
 shuffled=json.loads((out/"shuffled.json").read_text())
 models=call(a.base_url.rstrip("/")+"/models",key=a.api_key,timeout=a.timeout).get("data",[])
 ids=[m["id"] for m in models if m.get("id")]
 model=a.model or next((x for x in ids if "gpt-oss" in x.lower()),ids[0] if ids else None)
 if not model: raise RuntimeError("No model loaded at the API endpoint.")
 results={"experiment":"E46-A","model":model,"conditions":{}}
 for condition in ["no_memory","raw_text_memory","shuffled_mismatched_memory"]:
  rows=[] if condition=="no_memory" else (facts if condition=="raw_text_memory" else shuffled)
  memory="No facts available." if not rows else "\n".join(
   x["key"]+": color "+x["color"]+"; instrument "+x["instrument"]+"; region "+x["region"]+"."
   for x in rows)
  questions="\n".join(q["id"]+": "+q["question"] for q in qs)
  prompt=("Use only MEMORY. Missing facts must be answered UNKNOWN; never guess. "
   "Return only JSON as {\\"answers\\":{\\"ID\\":\\"answer\\"}}.\nMEMORY:\n"+memory+"\nQUESTIONS:\n"+questions)
  body={"model":model,"temperature":0,"max_tokens":1800,"stream":False,
   "messages":[{"role":"user","content":prompt}]}
  t=time.time(); response=call(a.base_url.rstrip("/")+"/chat/completions",body,a.api_key,a.timeout)
  elapsed=time.time()-t
  text=response["choices"][0]["message"]["content"]
  start=text.find("{"); end=text.rfind("}")
  parsed=json.loads(text[start:end+1])
  answers=parsed.get("answers",parsed)
  items=[]; correct=0; groups={}
  for q in qs:
   got=str(answers.get(q["id"],"")).strip().lower().strip(" .,!;:")
   expected=q["answer"].lower()
   ok=got==expected; correct+=int(ok)
   g=groups.setdefault(q["kind"],[0,0]); g[0]+=int(ok); g[1]+=1
   items.append(dict(id=q["id"],kind=q["kind"],expected=q["answer"],actual=got,correct=ok))
  results["conditions"][condition]=dict(correct=correct,total=len(qs),accuracy=correct/len(qs),
   by_kind={k:dict(correct=v[0],total=v[1],accuracy=v[0]/v[1]) for k,v in groups.items()},
   latency_seconds=round(elapsed,2),items=items,raw_response=text)
  print(condition+": "+str(correct)+"/"+str(len(qs)))
 results["storage_bytes"]={"raw_text":(out/"raw_memory.txt").stat().st_size,
  "structured_json":(out/"facts.json").stat().st_size}
 save(out/"evaluation.json",results)
 print("Saved",out/"evaluation.json")

def main():
 p=argparse.ArgumentParser()
 p.add_argument("command",choices=["prepare","evaluate"])
 p.add_argument("--output",type=Path,default=OUT)
 p.add_argument("--base-url",default="http://localhost:1234/v1")
 p.add_argument("--model",default=None)
 p.add_argument("--api-key",default=os.environ.get("LM_API_TOKEN"))
 p.add_argument("--timeout",type=int,default=300)
 a=p.parse_args()
 if a.command=="prepare": prepare(a.output)
 else: evaluate(a)

if __name__=="__main__":
 main()
