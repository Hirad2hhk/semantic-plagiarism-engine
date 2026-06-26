import random
import hashlib
from math import perm
from typing import Dict, List


class MinHashEngine:
    def __init__(self):
        self.hash_functions = []
        self.prime = 4294967311
        max_val = 2**32 - 1
        rng = random.Random(42)
        for _ in range(128):
            a = rng.randint(1, max_val)
            b = rng.randint(0, max_val)
            self.hash_functions.append((a, b))
            
    def _stable_hash(self, shingle: tuple) -> int:
         """Converts a tuple of strings into a stable 32-bit integer identifier."""
         shingle_bytes = str(shingle).encode('utf-8')
         hash_hex = hashlib.sha256(shingle_bytes).hexdigest()
         return int(hash_hex[:8], 16)
         
    def compute_signature(self, shingles: List[tuple]) -> List[int]:
        """Compute minhash signature for given document shards"""
        max_val = 2**32 - 1
        signature = [max_val] * len(self.hash_functions)

        if not shingles:
            return [max_val] * len(self.hash_functions)

        for shingle in shingles:
            shingle_hash = abs(self._stable_hash(shingle))
            for i, (a, b) in enumerate(self.hash_functions):
                perm_hash = (a * shingle_hash + b) % self.prime
                if perm_hash < signature[i]:
                    signature[i] = perm_hash
        return signature

    def estimate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity between two signatures"""
        if len(sig1) == 0 or len(sig2)==0:
            return 0.0
        matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
        return matches / len(sig1)
