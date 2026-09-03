"""E40 adaptive residual codec: a real model-coordination architecture."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import struct, zlib
MAGIC=b"NC40"; VERSION=1
class Predictor(Protocol):
    def predict(self, context: bytes, length: int, position: int, representation: "Representation") -> bytes: ...
@dataclass(frozen=True)
class RegionSignature:
    entropy_q:int; repetition_q:int; unique_q:int
@dataclass(frozen=True)
class Representation:
    name:str; context_size:int; compression_level:int
REPRESENTATIONS=(Representation("compact",8,1),Representation("balanced",16,6),Representation("deep",32,9))
class PreviousBytePredictor:
    def predict(self, context, length, position, representation):
        value=context[-1] if context else 0
        return bytes([value])*length
def analyze_region(data):
    if not data:return RegionSignature(0,255,0)
    counts=[0]*256
    for b in data:counts[b]+=1
    import math
    n=len(data); e=-sum((c/n)*math.log2(c/n) for c in counts if c)
    r=sum(a==b for a,b in zip(data,data[1:]))/max(1,n-1)
    u=sum(c>0 for c in counts)/256
    return RegionSignature(round(e/8*255),round(r*255),round(u*255))
def _xor(a,b):return bytes(x^y for x,y in zip(a,b))
def _predict(p,previous,length,position,rep):
    q=p.predict(previous[-rep.context_size:],length,position,rep)
    if len(q)!=length:raise ValueError("predictor returned incorrect region length")
    return q
def encode(data,predictor=None,region_size=1024):
    if region_size<=0:raise ValueError("region_size must be positive")
    predictor=predictor or PreviousBytePredictor(); out=bytearray(MAGIC+struct.pack("<BII",VERSION,region_size,(len(data)+region_size-1)//region_size)); previous=b""
    for pos in range(0,len(data),region_size):
        region=data[pos:pos+region_size]; sig=analyze_region(region); candidates=[]
        for rid,rep in enumerate(REPRESENTATIONS):
            residual=_xor(region,_predict(predictor,previous,len(region),pos//region_size,rep)); coded=zlib.compress(residual,rep.compression_level)
            candidates.append((9+len(coded),rid,coded,len(residual)))
        _,rid,coded,raw=min(candidates,key=lambda x:(x[0],x[1])); out+=struct.pack("<BII",rid,len(coded),raw)+coded; previous=region
    return bytes(out)
def decode(blob,predictor=None):
    predictor=predictor or PreviousBytePredictor()
    if len(blob)<13 or blob[:4]!=MAGIC:raise ValueError("invalid NC40 stream")
    version,region_size,count=struct.unpack_from("<BII",blob,4)
    if version!=VERSION:raise ValueError("unsupported NC40 version")
    p=13; out=bytearray(); previous=b""
    for pos in range(count):
        if p+9>len(blob):raise ValueError("truncated region header")
        rid,cl,raw=struct.unpack_from("<BII",blob,p);p+=9
        if rid>=len(REPRESENTATIONS) or p+cl>len(blob):raise ValueError("invalid/truncated region")
        residual=zlib.decompress(blob[p:p+cl]);p+=cl
        if len(residual)!=raw:raise ValueError("residual length mismatch")
        region=_xor(residual,_predict(predictor,previous,raw,pos,REPRESENTATIONS[rid]));out+=region;previous=region
    if p!=len(blob):raise ValueError("trailing bytes")
    return bytes(out)
def roundtrip(data,predictor=None,region_size=1024):
    blob=encode(data,predictor,region_size); decoded=decode(blob,predictor)
    if decoded!=data:raise AssertionError("exact reconstruction failed")
    return blob,len(blob)/max(1,len(data))
if __name__=="__main__":
    x=b"neural compression: persistent structure. "*1000; b,r=roundtrip(x); print(f"input={len(x)} encoded={len(b)} ratio={r:.4f} exact=True")
