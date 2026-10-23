import re, hashlib, math
from typing import List

class OfflineEmbedding:
    """Deterministic lexical baseline: hashed character 3-grams + words, 384D."""
    def __init__(self, dim=384): self.dim=dim
    def name(self): return "offline-hashed-lexical-384"
    def __call__(self, input: List[str]) -> List[List[float]]: return [self.embed(x) for x in input]
    def embed_documents(self, input): return self(input)
    def embed_query(self, input): return self(input)
    def embed(self, text):
        v=[0.0]*self.dim; s=text.lower()
        feats=[s[i:i+3] for i in range(max(0,len(s)-2))]+re.findall(r"[\wåäö]+",s)
        for f in feats:
            h=int.from_bytes(hashlib.blake2b(f.encode(),digest_size=8).digest(),'little'); v[h%self.dim] += -1.0 if h>>63 else 1.0
        n=math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/n for x in v]
