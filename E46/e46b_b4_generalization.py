#!/usr/bin/env python3
"""E46-B4: entity-disjoint retrieval generalization and memory scaling.

Qwen is a frozen activation encoder. Only the adapter's address_writer and
query_encoder are trained. The test entities never contribute training gradients
or checkpoint-selection metrics. The final test reports retrieval over unseen
entities at several candidate-bank sizes.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor
from E46.e46b_memory_adapter import NeuralMemoryAdapter

CATEGORIES = [
    ("vessel", "ORBIT", "calibration temperature", "degrees Celsius",
     ["What temperature is logged for {entity}?", "Retrieve the calibration reading belonging to {entity}.", "Which temperature should be associated with vessel {entity}?"],
     ["Give me the recorded thermal calibration for {entity}.", "What is {entity}'s calibration temperature?"]),
    ("instrument", "INSTR", "manufacture year", "",
     ["When was instrument {entity} manufactured?", "Find the production year for {entity}.", "Which year belongs to instrument {entity}?"],
     ["Tell me when {entity} was made.", "What is the manufacturing date year for {entity}?"]),
    ("rover", "ROVER", "rover designation", "",
     ["What is the name assigned to rover {entity}?", "Identify the designation of {entity}.", "Which callsign belongs to rover {entity}?"],
     ["Tell me what this rover is called: {entity}.", "Retrieve the assigned name for {entity}."]),
    ("sensor", "SENSOR", "installation sector", "",
     ["Where is sensor {entity} installed?", "Find the deployment sector for {entity}.", "Which area contains sensor {entity}?"],
     ["Tell me the installation location of {entity}.", "In which sector can {entity} be found?"]),
    ("drone", "DRONE", "battery capacity", "watt-hours",
     ["What battery capacity is recorded for drone {entity}?", "How much energy can {entity}'s battery store?", "Retrieve the power-storage rating for {entity}."],
     ["Give me the battery size of {entity} in watt-hours.", "What is the energy capacity assigned to {entity}?"]),
    ("archive", "ARCH", "catalogue count", "entries",
     ["How many entries are listed in archive {entity}?", "Find the catalogue total for {entity}.", "What is the record count of {entity}?"],
     ["Tell me the number of catalogued items in {entity}.", "How many records does {entity} contain?"]),
    ("station", "STATION", "opening year", "",
     ["When did station {entity} begin operation?", "What year was {entity} opened?", "Retrieve the launch year for {entity}."],
     ["In which year did {entity} become operational?", "Tell me the opening year associated with {entity}."]),
    ("greenhouse", "HOUSE", "crop variety", "",
     ["Which crop variety is growing in greenhouse {entity}?", "Find the planted variety for {entity}.", "What crop is assigned to greenhouse {entity}?"],
     ["Tell me which variety is cultivated in {entity}.", "What is growing in greenhouse {entity}?"]),
]
WORDS = ["Kestrel","Juniper","Cinder","Morrow","Solace","Thistle","Ember","Quartz",
         "Willow","Copper","Lumen","Aster","Mica","Saffron","Cobalt","Pollen",
         "Tundra","Vesper","Clover","Onyx","Bramble","Nimbus","Fable","Sable",
         "Comet","Mosaic","Cypress","Topaz","Rill","Marble","Flint","Auburn"]
SECTORS = ["eastern greenhouse","north ridge","lower valley","west platform","river annex","upper dome","central bay","southern field"]
CROPS = ["purple basil","dwarf kale","red amaranth","lemon thyme","golden beet","blue corn","winter spinach","striped tomato"]

def make_spec(seed=4604):
    rng = random.Random(seed)
    facts = []
    for category, prefix, attr, unit, train_templates, test_templates in CATEGORIES:
        for i in range(32):
            entity = f"{prefix}-{i+1:03d}"
            if category == "vessel": value, fact = 10 + ((i * 7 + 3) % 61), None
            elif category == "instrument": value = 1995 + ((i * 11 + 4) % 51)
            elif category == "rover": value = WORDS[(i * 7 + 2) % len(WORDS)]
            elif category == "sensor": value = SECTORS[(i * 3 + 1) % len(SECTORS)]
            elif category == "drone": value = 180 + ((i * 83 + 37) % 1200)
            elif category == "archive": value = 35 + ((i * 47 + 19) % 900)
            elif category == "station": value = 2020 + ((i * 3 + 1) % 45)
            else: value = CROPS[(i * 5 + 3) % len(CROPS)]
            if category == "vessel":
                fact = f"Vessel {entity} has a calibration temperature of {value} degrees Celsius."
            elif category == "instrument": fact = f"Instrument {entity} was manufactured in {value}."
            elif category == "rover": fact = f"Rover {entity} is named {value}."
            elif category == "sensor": fact = f"Sensor {entity} is installed in the {value}."
            elif category == "drone": fact = f"Drone {entity} has a battery capacity of {value} watt-hours."
            elif category == "archive": fact = f"Archive {entity} contains {value} catalogued entries."
            elif category == "station": fact = f"Station {entity} opened in {value}."
            else: fact = f"Greenhouse {entity} is growing {value}."
            queries = [t.format(entity=entity) for t in train_templates]
            heldout = [t.format(entity=entity) for t in test_templates]
            facts.append({
                "id": f"{category}_{i+1:03d}", "category": category,
                "entity": entity, "value": value, "fact": fact,
                "capture_prompt": f"Remember this record: {fact}",
                "split": "train_entity" if i < 24 else "unseen_entity",
                "train_queries": queries[:2] if i < 24 else [],
                "validation_queries": [queries[2]] if i < 24 else [],
                "test_queries": heldout if i >= 24 else [],
            })
    rng.shuffle(facts)
    return {"format_version": 1, "experiment": "E46-B4", "seed": seed,
            "design": "8 categories x 32 entities; 24 train entities and 8 unseen entities per category",
            "facts": facts}

def host(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = Path(model_path).expanduser()
    if not p.is_dir() or not (p / "config.json").is_file():
        raise FileNotFoundError(f"Expected local Transformers model directory with config.json: {p}")
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(p), dtype=torch.float32,
        local_files_only=True, trust_remote_code=False).to(device).eval()
    model.config.use_cache = False
    return tok, model

def hidden(output):
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor): return output[0]
    if isinstance(output, Tensor): return output
    raise TypeError(f"Unsupported block output: {type(output)!r}")

def collect(args):
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda")
    tok, model = host(args.model_path, device)
    names = [n for n,_ in model.named_modules() if re.search(r"(?:^|\.)(?:layers|blocks|h)\.\d+$", n)]
    if not names: raise RuntimeError("Could not find decoder block candidates.")
    layer_path = args.layer_path or names[len(names)//2]
    cache = {}
    def hook(_m,_i,out): cache["v"] = hidden(out).detach().float().mean(1).squeeze(0).cpu()
    handle = model.get_submodule(layer_path).register_forward_hook(hook)
    spec = json.loads(Path(args.spec).read_text())
    rows = []
    def encode(prompt):
        cache.clear()
        with torch.inference_mode():
            model(**tok(prompt, return_tensors="pt").to(device), use_cache=False)
        if "v" not in cache: raise RuntimeError(f"Layer {layer_path} was not called.")
        return cache["v"].clone()
    for j,f in enumerate(spec["facts"],1):
        row = {k:f[k] for k in ("id","category","entity","value","fact","split","capture_prompt")}
        row["capture"] = encode(f["capture_prompt"])
        for key in ("train_queries","validation_queries","test_queries"):
            row[key] = [{"text":q,"vector":encode(q)} for q in f[key]]
        rows.append(row)
        if j % 32 == 0: print(f"captured {j}/{len(spec['facts'])} facts", flush=True)
    handle.remove()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"format_version":1,"model_path":str(Path(args.model_path).expanduser()),
        "model_type":getattr(model.config,"model_type","unknown"),
        "hidden_dim":int(model.config.hidden_size),"layer_path":layer_path,"facts":rows},out)
    print(json.dumps({"phase":"collect","facts":len(rows),
        "train_entities":sum(x["split"]=="train_entity" for x in rows),
        "unseen_entities":sum(x["split"]=="unseen_entity" for x in rows),
        "hidden_dim":int(model.config.hidden_size),"layer_path":layer_path,
        "device":str(device),"dataset":str(out)},indent=2))

def load_data(path):
    d=torch.load(path,map_location="cpu",weights_only=False)
    if d.get("format_version")!=1 or len(d.get("facts",[]))<16:
        raise ValueError("Invalid or incomplete E46-B4 activation dataset.")
    return d

def score_queries(adapter, memories, queries, candidate_indices=None):
    if not queries: return {"queries":0,"top1_accuracy":None,"mean_reciprocal_rank":None,"mean_margin":None}
    mem=torch.stack([memories[i]["capture"].float() for i in range(len(memories))])
    with torch.no_grad(): addr=F.normalize(adapter.address_writer(mem),dim=-1)
    qvec=torch.stack([q["vector"].float() for q in queries])
    labels=torch.tensor([q["target"] for q in queries],dtype=torch.long)
    with torch.no_grad(): q=F.normalize(adapter.query_encoder(qvec),dim=-1)
    if candidate_indices is None: candidate_indices=list(range(len(memories)))
    positions={idx:p for p,idx in enumerate(candidate_indices)}
    keep=torch.tensor([int(y) in positions for y in labels])
    q=q[keep]; labels=labels[keep]
    if not len(labels): return {"queries":0,"top1_accuracy":None,"mean_reciprocal_rank":None,"mean_margin":None}
    local_labels=torch.tensor([positions[int(y)] for y in labels])
    sims=q @ addr[candidate_indices].T
    order=sims.argsort(dim=-1,descending=True)
    ranks=(order==local_labels[:,None]).nonzero()[:,1]+1
    top=sims.topk(min(2,sims.shape[1]),dim=-1).values
    pred=sims.argmax(-1)
    return {"queries":int(len(labels)),"top1_accuracy":float((pred==local_labels).float().mean()),
        "mean_reciprocal_rank":float((1.0/ranks.float()).mean()),
        "mean_margin":float((sims.gather(1,local_labels[:,None]).squeeze(1)-top[:,-1]).mean()),
        "mean_positive_rank":float(ranks.float().mean())}

def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    d=load_data(args.dataset); facts=d["facts"]
    train_idx=[i for i,f in enumerate(facts) if f["split"]=="train_entity"]
    unseen_idx=[i for i,f in enumerate(facts) if f["split"]=="unseen_entity"]
    if len(train_idx)<16 or len(unseen_idx)<8: raise ValueError("Need disjoint train and unseen entities.")
    train_facts=[facts[i] for i in train_idx]
    captures=torch.stack([f["capture"].float() for f in train_facts])
    qvec=[]; labels=[]
    for local,f in enumerate(train_facts):
        for q in f["train_queries"]: qvec.append(q["vector"].float()); labels.append(local)
    queries=torch.stack(qvec); targets=torch.tensor(labels)
    adapter=NeuralMemoryAdapter(int(d["hidden_dim"]),args.address_dim,args.payload_dim,top_k=1)
    params=list(adapter.address_writer.parameters())+list(adapter.query_encoder.parameters())
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
    def eval_split(split):
        qs=[]
        for idx in train_idx:
            for q in facts[idx][split]: qs.append({"vector":q["vector"],"target":train_idx.index(idx)})
        return score_queries(adapter,train_facts,qs)
    before=eval_split("validation_queries")
    best=-1; best_state=None; best_epoch=0; history=[]
    for epoch in range(1,args.epochs+1):
        adapter.train(); opt.zero_grad(set_to_none=True)
        a=F.normalize(adapter.address_writer(captures),dim=-1)
        q=F.normalize(adapter.query_encoder(queries),dim=-1)
        loss=F.cross_entropy((q@a.T)/args.temperature,targets)
        loss.backward(); opt.step(); adapter.eval()
        val=eval_split("validation_queries")
        metric=val["top1_accuracy"] or 0.0
        history.append({"epoch":epoch,"loss":float(loss.item()),"validation_top1":metric})
        if metric>best:
            best=metric; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        if epoch==1 or epoch%args.log_every==0 or epoch==args.epochs:
            print(f"epoch={epoch:04d} loss={loss.item():.4f} validation_top1={metric:.3f}",flush=True)
    adapter.load_state_dict(best_state); adapter.eval()
    # Full bank includes both seen and unseen entity captures; no test query is used in training.
    adapter.bank = adapter.bank.__class__(adapter.address_dim,adapter.payload_dim)
    for f in facts:
        adapter.write(f["capture"].float().unsqueeze(0),{
            "capture_id":f["id"],"entity":f["entity"],"category":f["category"],
            "split":f["split"],"fact":f["fact"],"model_type":d["model_type"],
            "layer_path":d["layer_path"]})
    out=Path(args.adapter_dir); adapter.save(out)
    unseen_mem=[facts[i] for i in unseen_idx]
    test_q=[]
    for i in unseen_idx:
        for q in facts[i]["test_queries"]: test_q.append({"vector":q["vector"],"target":i})
    # All 256 records is the primary generalization test. Smaller banks are sampled
    # deterministically while always retaining the correct target memory.
    rng=random.Random(args.seed+9000); scale={}
    for size in (8,32,128,256):
        size=min(size,len(facts)); rows=[]
        for q in test_q:
            distractors=[i for i in range(len(facts)) if i!=q["target"]]
            chosen=rng.sample(distractors,size-1)+[q["target"]]
            rows.append((q,chosen))
        # Aggregate one query at a time so each query has its own candidate set.
        vals=[]
        for q,chosen in rows:
            m=torch.stack([facts[i]["capture"].float() for i in chosen])
            v=q["vector"].float().unsqueeze(0)
            with torch.no_grad():
                ma=F.normalize(adapter.address_writer(m),dim=-1)
                qa=F.normalize(adapter.query_encoder(v),dim=-1)
                sim=qa@ma.T
            pos=chosen.index(q["target"]); order=sim.argsort(-1,descending=True)
            rank=int((order==pos).nonzero()[0,1])+1
            sortedv=sim.sort(descending=True).values[0]
            vals.append((rank,float(sim[0,pos]-sortedv[1])))
        scale[str(size)]={"queries":len(vals),"top1_accuracy":sum(x[0]==1 for x in vals)/len(vals),
            "mean_positive_rank":sum(x[0] for x in vals)/len(vals),
            "mean_margin":sum(x[1] for x in vals)/len(vals)}
    full=score_queries(adapter,facts,test_q)
    # Also evaluate unseen facts against only the unseen-entity sub-bank.
    unseen_queries=[{"vector":q["vector"],"target":unseen_idx.index(i)}
        for i in unseen_idx for q in facts[i]["test_queries"]]
    unseen_only=score_queries(adapter,unseen_mem,unseen_queries)
    report={"experiment":"E46-B4","seed":args.seed,"facts":len(facts),
        "training_entities":len(train_idx),"unseen_entities":len(unseen_idx),
        "training_queries":len(queries),"validation_queries":sum(len(facts[i]["validation_queries"]) for i in train_idx),
        "unseen_test_queries":len(test_q),"trainable_components":["address_writer","query_encoder"],
        "frozen_components":["Qwen host","payload_writer","payload_reader"],
        "best_epoch":best_epoch,"validation_before_training":before,
        "validation_after_training":eval_split("validation_queries"),
        "unseen_entity_full_bank_256":full,"unseen_entity_only_bank_64":unseen_only,
        "unseen_entity_bank_scale_sweep":scale,"adapter_dir":str(out),
        "note":"Unseen entity identifiers and their queries are excluded from gradient updates and checkpoint selection. This is controlled synthetic generalization, not open-domain semantic generalization."}
    (out/"e46b4_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    g=sub.add_parser("generate-spec"); g.add_argument("--output",default="E46/e46b_b4_dataset.json"); g.add_argument("--seed",type=int,default=4604)
    c=sub.add_parser("collect"); c.add_argument("--model-path",required=True); c.add_argument("--spec",default="E46/e46b_b4_dataset.json"); c.add_argument("--output",default="runs/E46B/B4/activations.pt"); c.add_argument("--layer-path",default=None); c.add_argument("--device",choices=["auto","cpu"],default="cpu")
    t=sub.add_parser("train"); t.add_argument("--dataset",default="runs/E46B/B4/activations.pt"); t.add_argument("--adapter-dir",default="runs/E46B/B4/adapter"); t.add_argument("--address-dim",type=int,default=128); t.add_argument("--payload-dim",type=int,default=128); t.add_argument("--epochs",type=int,default=400); t.add_argument("--lr",type=float,default=1e-3); t.add_argument("--weight-decay",type=float,default=1e-4); t.add_argument("--temperature",type=float,default=.07); t.add_argument("--seed",type=int,default=46); t.add_argument("--log-every",type=int,default=25)
    g.set_defaults(func=lambda a: generate_spec(a)); c.set_defaults(func=collect); t.set_defaults(func=train)
    args=p.parse_args()
    if args.cmd=="generate-spec":
        spec=make_spec(args.seed); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(spec,indent=2))
        print(json.dumps({"output":str(out),"facts":len(spec["facts"]),"train_entities":sum(x["split"]=="train_entity" for x in spec["facts"]),"unseen_entities":sum(x["split"]=="unseen_entity" for x in spec["facts"])},indent=2))
    else: args.func(args)

if __name__=="__main__": main()
