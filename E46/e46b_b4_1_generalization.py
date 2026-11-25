#!/usr/bin/env python3
"""E46-B4.1: factorized entity/wording generalization and nested-bank scaling."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from torch import Tensor
from E46.e46b_memory_adapter import NeuralMemoryAdapter

CATEGORIES=[
("vessel","ORBIT",["What temperature is logged for {entity}?","Retrieve the calibration reading belonging to {entity}.","Which temperature should be associated with vessel {entity}?"],["Give me the recorded thermal calibration for {entity}.","What is {entity}'s calibration temperature?"]),
("instrument","INSTR",["When was instrument {entity} manufactured?","Find the production year for {entity}.","Which year belongs to instrument {entity}?"],["Tell me when {entity} was made.","What is the manufacturing date year for {entity}?"]),
("rover","ROVER",["What is the name assigned to rover {entity}?","Identify the designation of {entity}.","Which callsign belongs to {entity}?"],["Tell me what this rover is called: {entity}.","Retrieve the assigned name for {entity}."]),
("sensor","SENSOR",["Where is sensor {entity} installed?","Find the deployment sector for {entity}.","Which area contains sensor {entity}?"],["Tell me the installation location of {entity}.","In which sector can {entity} be found?"]),
("drone","DRONE",["What battery capacity is recorded for drone {entity}?","How much energy can {entity}'s battery store?","Retrieve the power-storage rating for {entity}."],["Give me the battery size of {entity} in watt-hours.","What is the energy capacity assigned to {entity}?"]),
("archive","ARCH",["How many entries are listed in archive {entity}?","Find the catalogue total for {entity}.","What is the record count of {entity}?"],["Tell me the number of catalogued items in {entity}.","How many records does {entity} contain?"]),
("station","STATION",["When did station {entity} begin operation?","What year was {entity} opened?","Retrieve the launch year for {entity}."],["In which year did {entity} become operational.","Tell me the opening year associated with {entity}."]),
("greenhouse","HOUSE",["Which crop variety is growing in greenhouse {entity}?","Find the planted variety for {entity}.","What crop is assigned to greenhouse {entity}?"],["Tell me which variety is cultivated in {entity}.","What is growing in greenhouse {entity}?"]),
]
WORDS=["Kestrel","Juniper","Cinder","Morrow","Solace","Thistle","Ember","Quartz","Willow","Copper","Lumen","Aster","Mica","Saffron","Cobalt","Pollen","Tundra","Vesper","Clover","Onyx","Bramble","Nimbus","Fable","Sable","Comet","Mosaic","Cypress","Topaz","Rill","Marble","Flint","Auburn"]
SECTORS=["eastern greenhouse","north ridge","lower valley","west platform","river annex","upper dome","central bay","southern field"]
CROPS=["purple basil","dwarf kale","red amaranth","lemon thyme","golden beet","blue corn","winter spinach","striped tomato"]

def make_spec(seed=4614):
    rng=random.Random(seed); facts=[]
    for category,prefix,familiar,novel in CATEGORIES:
        unseen=set(rng.sample(range(32),8))
        for i in range(32):
            entity=f"{prefix}-{i+1:03d}"
            if category=="vessel": value=10+((i*7+3)%61)
            elif category=="instrument": value=1995+((i*11+4)%51)
            elif category=="rover": value=WORDS[(i*7+2)%len(WORDS)]
            elif category=="sensor": value=SECTORS[(i*3+1)%len(SECTORS)]
            elif category=="drone": value=180+((i*83+37)%1200)
            elif category=="archive": value=35+((i*47+19)%900)
            elif category=="station": value=2020+((i*3+1)%45)
            else: value=CROPS[(i*5+3)%len(CROPS)]
            if category=="vessel": fact=f"Vessel {entity} has a calibration temperature of {value} degrees Celsius."
            elif category=="instrument": fact=f"Instrument {entity} was manufactured in {value}."
            elif category=="rover": fact=f"Rover {entity} is named {value}."
            elif category=="sensor": fact=f"Sensor {entity} is installed in the {value}."
            elif category=="drone": fact=f"Drone {entity} has a battery capacity of {value} watt-hours."
            elif category=="archive": fact=f"Archive {entity} contains {value} catalogued entries."
            elif category=="station": fact=f"Station {entity} opened in {value}."
            else: fact=f"Greenhouse {entity} is growing {value}."
            fmt=lambda ts:[t.format(entity=entity) for t in ts]
            seen=i not in unseen
            facts.append({"id":f"{category}_{i+1:03d}","category":category,"entity":entity,
                "value":value,"fact":fact,"capture_prompt":f"Remember this record: {fact}",
                "split":"train_entity" if seen else "unseen_entity",
                "train_queries":fmt(familiar[:2]) if seen else [],
                "validation_queries":fmt(familiar[2:]) if seen else [],
                "seen_novel_queries":fmt(novel) if seen else [],
                "unseen_familiar_queries":fmt(familiar[:2]) if not seen else [],
                "unseen_novel_queries":fmt(novel) if not seen else []})
    rng.shuffle(facts)
    return {"format_version":2,"experiment":"E46-B4.1","seed":seed,
      "design":"8 categories x 32 entities; randomized stratified 24/8 split; factorized entity/wording tests","facts":facts}

def host(model_path,device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    p=Path(model_path).expanduser()
    if not p.is_dir() or not (p/"config.json").is_file(): raise FileNotFoundError(f"Expected local Transformers model directory: {p}")
    tok=AutoTokenizer.from_pretrained(str(p),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(p),dtype=torch.float32,local_files_only=True,trust_remote_code=False).to(device).eval()
    model.config.use_cache=False
    return tok,model

def hidden(output):
    if isinstance(output,tuple) and output and isinstance(output[0],Tensor): return output[0]
    if isinstance(output,Tensor): return output
    raise TypeError(f"Unsupported block output: {type(output)!r}")

def collect(args):
    device=torch.device("cpu" if args.device=="cpu" or not torch.cuda.is_available() else "cuda")
    tok,model=host(args.model_path,device)
    names=[n for n,_ in model.named_modules() if re.search(r"(?:^|\.)(?:layers|blocks|h)\.\d+$",n)]
    if not names: raise RuntimeError("Could not find decoder block candidates.")
    layer_path=args.layer_path or names[len(names)//2]; cache={}
    def hook(_m,_i,out): cache["v"]=hidden(out).detach().float().mean(1).squeeze(0).cpu()
    handle=model.get_submodule(layer_path).register_forward_hook(hook)
    spec=json.loads(Path(args.spec).read_text()); rows=[]
    keys=("train_queries","validation_queries","seen_novel_queries","unseen_familiar_queries","unseen_novel_queries")
    def encode(prompt):
        cache.clear()
        with torch.inference_mode(): model(**tok(prompt,return_tensors="pt").to(device),use_cache=False)
        if "v" not in cache: raise RuntimeError(f"Layer {layer_path} was not called.")
        return cache["v"].clone()
    for j,f in enumerate(spec["facts"],1):
        row={k:f[k] for k in ("id","category","entity","value","fact","split","capture_prompt")}
        row["capture"]=encode(f["capture_prompt"])
        for key in keys: row[key]=[{"text":q,"vector":encode(q)} for q in f[key]]
        rows.append(row)
        if j%32==0: print(f"captured {j}/{len(spec['facts'])} facts",flush=True)
    handle.remove(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"format_version":2,"model_path":str(Path(args.model_path).expanduser()),
      "model_type":getattr(model.config,"model_type","unknown"),"hidden_dim":int(model.config.hidden_size),
      "layer_path":layer_path,"facts":rows},out)
    print(json.dumps({"phase":"collect","facts":len(rows),"train_entities":sum(x["split"]=="train_entity" for x in rows),
      "unseen_entities":sum(x["split"]=="unseen_entity" for x in rows),"hidden_dim":int(model.config.hidden_size),
      "layer_path":layer_path,"device":str(device),"dataset":str(out)},indent=2))

def load_data(path):
    d=torch.load(path,map_location="cpu",weights_only=False)
    if d.get("format_version")!=2 or len(d.get("facts",[]))!=256: raise ValueError("Invalid/incomplete B4.1 activation dataset.")
    return d

def query_rows(facts,indices,key):
    return [{"vector":q["vector"],"target":i,"category":facts[i]["category"]}
      for i in indices for q in facts[i][key]]

def score_queries(adapter,facts,queries,candidate_indices=None):
    empty={"queries":0,"top1_accuracy":None,"mean_reciprocal_rank":None,"mean_positive_rank":None,"mean_margin":None}
    if not queries: return empty
    if candidate_indices is None: candidate_indices=list(range(len(facts)))
    positions={idx:p for p,idx in enumerate(candidate_indices)}
    valid=[q for q in queries if q["target"] in positions]
    if not valid: return empty
    mem=torch.stack([facts[i]["capture"].float() for i in candidate_indices])
    qvec=torch.stack([q["vector"].float() for q in valid])
    labels=torch.tensor([positions[q["target"]] for q in valid],dtype=torch.long)
    with torch.no_grad():
        addr=F.normalize(adapter.address_writer(mem),dim=-1)
        q=F.normalize(adapter.query_encoder(qvec),dim=-1)
        sims=q@addr.T
    order=sims.argsort(dim=-1,descending=True)
    ranks=(order==labels[:,None]).nonzero()[:,1]+1
    positive=sims.gather(1,labels[:,None]).squeeze(1)
    negative=sims.masked_fill(F.one_hot(labels,num_classes=sims.shape[1]).bool(),-torch.inf)
    margins=positive-negative.max(dim=-1).values
    result={"queries":len(valid),"top1_accuracy":float((ranks==1).float().mean()),
      "mean_reciprocal_rank":float((1/ranks.float()).mean()),"mean_positive_rank":float(ranks.float().mean()),
      "mean_margin":float(margins.mean())}
    result["by_category"]={}
    for cat in sorted({x["category"] for x in valid}):
        ix=[j for j,x in enumerate(valid) if x["category"]==cat]; rr=ranks[ix]
        result["by_category"][cat]={"queries":len(ix),"top1_accuracy":float((rr==1).float().mean()),
          "mean_reciprocal_rank":float((1/rr.float()).mean()),"mean_positive_rank":float(rr.float().mean()),
          "mean_margin":float(margins[ix].mean())}
    return result

def nested_scale(adapter,facts,queries,seed):
    rng=random.Random(seed); orders={}
    for q in queries:
        distractors=[i for i in range(len(facts)) if i!=q["target"]]; rng.shuffle(distractors)
        orders[id(q)]=[q["target"]]+distractors
    out={}
    for size in (8,32,128,256):
        vals=[]
        for q in queries:
            chosen=orders[id(q)][:size]
            m=torch.stack([facts[i]["capture"].float() for i in chosen])
            with torch.no_grad():
                a=F.normalize(adapter.address_writer(m),dim=-1)
                v=F.normalize(adapter.query_encoder(q["vector"].float().unsqueeze(0)),dim=-1)
                sim=v@a.T
            pos=chosen.index(q["target"]); neg=sim.clone(); neg[0,pos]=-torch.inf
            rank=int((sim.argsort(-1,descending=True)==pos).nonzero()[0,1])+1
            vals.append((rank,float(sim[0,pos]-neg.max())))
        out[str(size)]={"queries":len(vals),"top1_accuracy":sum(x[0]==1 for x in vals)/len(vals),
          "mean_positive_rank":sum(x[0] for x in vals)/len(vals),"mean_margin":sum(x[1] for x in vals)/len(vals)}
    return out

def train_one(d,args,seed):
    torch.manual_seed(seed); random.seed(seed); facts=d["facts"]
    train_idx=[i for i,f in enumerate(facts) if f["split"]=="train_entity"]
    unseen_idx=[i for i,f in enumerate(facts) if f["split"]=="unseen_entity"]
    train_facts=[facts[i] for i in train_idx]
    captures=torch.stack([f["capture"].float() for f in train_facts]); qvec=[]; labels=[]
    for local,f in enumerate(train_facts):
        for q in f["train_queries"]: qvec.append(q["vector"].float()); labels.append(local)
    queries=torch.stack(qvec); targets=torch.tensor(labels)
    adapter=NeuralMemoryAdapter(int(d["hidden_dim"]),args.address_dim,args.payload_dim,top_k=1)
    params=list(adapter.address_writer.parameters())+list(adapter.query_encoder.parameters())
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
    valq=query_rows(facts,train_idx,"validation_queries")
    before=score_queries(adapter,train_facts,valq); best=-1.; state=None; best_epoch=0
    for epoch in range(1,args.epochs+1):
        adapter.train(); opt.zero_grad(set_to_none=True)
        a=F.normalize(adapter.address_writer(captures),dim=-1); q=F.normalize(adapter.query_encoder(queries),dim=-1)
        loss=F.cross_entropy((q@a.T)/args.temperature,targets); loss.backward(); opt.step(); adapter.eval()
        val=score_queries(adapter,train_facts,valq); metric=val["top1_accuracy"] or 0.
        if metric>best:
            best=metric; best_epoch=epoch; state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        if epoch==1 or epoch%args.log_every==0 or epoch==args.epochs:
            print(f"seed={seed} epoch={epoch:04d} loss={loss.item():.4f} validation_top1={metric:.3f}",flush=True)
    adapter.load_state_dict(state); adapter.eval(); adapter.bank=adapter.bank.__class__(adapter.address_dim,adapter.payload_dim)
    for f in facts:
        adapter.write(f["capture"].float().unsqueeze(0),{"capture_id":f["id"],"entity":f["entity"],
          "category":f["category"],"split":f["split"],"fact":f["fact"],"model_type":d["model_type"],"layer_path":d["layer_path"]})
    out=Path(args.output_root)/f"seed_{seed}"; adapter.save(out)
    seen_novel=query_rows(facts,train_idx,"seen_novel_queries")
    unseen_familiar=query_rows(facts,unseen_idx,"unseen_familiar_queries")
    unseen_novel=query_rows(facts,unseen_idx,"unseen_novel_queries")
    return {"seed":seed,"best_epoch":best_epoch,"validation_before_training":before,
      "validation_after_training":score_queries(adapter,train_facts,valq),
      "seen_entity_novel_wording_full_bank":score_queries(adapter,facts,seen_novel),
      "unseen_entity_familiar_wording_full_bank":score_queries(adapter,facts,unseen_familiar),
      "unseen_entity_novel_wording_full_bank":score_queries(adapter,facts,unseen_novel),
      "unseen_entity_novel_wording_nested_bank_scale":nested_scale(adapter,facts,unseen_novel,seed+9000),
      "adapter_dir":str(out)}

def train(args):
    d=load_data(args.dataset); seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    results=[train_one(d,args,s) for s in seeds]
    report={"experiment":"E46-B4.1","dataset":str(args.dataset),"facts":len(d["facts"]),
      "training_entities":sum(f["split"]=="train_entity" for f in d["facts"]),
      "unseen_entities":sum(f["split"]=="unseen_entity" for f in d["facts"]),
      "training_queries":sum(len(f["train_queries"]) for f in d["facts"]),
      "trainable_components":["address_writer","query_encoder"],
      "frozen_components":["Qwen host","payload_writer","payload_reader"],
      "conditions":{"seen_entity_familiar_wording":"validation; checkpoint selection",
        "seen_entity_novel_wording":"held-out templates for trained entities",
        "unseen_entity_familiar_wording":"training-style templates applied to unseen entities",
        "unseen_entity_novel_wording":"both entity and wording unseen"},
      "seeds":results,"note":"Synthetic controlled addressing only; no payload reconstruction, answer utility, open-domain transfer, or compression benefit is established. Validation is used for checkpoint selection. Nested candidate sets use a fixed per-query distractor ordering and always include the target."}
    out=Path(args.report); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    g=sub.add_parser("generate-spec"); g.add_argument("--output",default="E46/e46b_b4_1_dataset.json"); g.add_argument("--seed",type=int,default=4614)
    c=sub.add_parser("collect"); c.add_argument("--model-path",required=True); c.add_argument("--spec",default="E46/e46b_b4_1_dataset.json"); c.add_argument("--output",default="runs/E46B/B41/activations.pt"); c.add_argument("--layer-path",default=None); c.add_argument("--device",choices=["auto","cpu"],default="cpu")
    t=sub.add_parser("train"); t.add_argument("--dataset",default="runs/E46B/B41/activations.pt"); t.add_argument("--output-root",default="runs/E46B/B41/adapters"); t.add_argument("--report",default="runs/E46B/B41/e46b4_1_report.json"); t.add_argument("--address-dim",type=int,default=128); t.add_argument("--payload-dim",type=int,default=128); t.add_argument("--epochs",type=int,default=400); t.add_argument("--lr",type=float,default=1e-3); t.add_argument("--weight-decay",type=float,default=1e-4); t.add_argument("--temperature",type=float,default=.07); t.add_argument("--seeds",default="46,47,48"); t.add_argument("--log-every",type=int,default=25)
    args=p.parse_args()
    if args.cmd=="generate-spec":
        spec=make_spec(args.seed); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(spec,indent=2))
        print(json.dumps({"output":str(out),"facts":len(spec["facts"]),"train_entities":sum(x["split"]=="train_entity" for x in spec["facts"]),"unseen_entities":sum(x["split"]=="unseen_entity" for x in spec["facts"])},indent=2))
    elif args.cmd=="collect": collect(args)
    else: train(args)
if __name__=="__main__": main()
