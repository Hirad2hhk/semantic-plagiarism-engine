import random
from math import perm
from typing import Dict, List


class MinHashEngine:
    def __init__(self):
        self.hash_functions = []
        self.hash_functions = []
        self.prime = 4294967311
        max_val = 2**32 - 1
        for _ in range(128):
            a = random.randint(1, max_val)
            b = random.randint(0, max_val)
            self.hash_functions.append((a, b))

    def compute_signature(self, shingles: List[tuple]) -> List[int]:
        """Compute minhash signature for given document shards"""
        signature = [int("inf")] * len(self.hash_functions)
        for shingle in shingles:
            shingle_hash = abs(hash(shingle))
            for i, (a, b) in enumerate(self.hash_functions):
                perm_hash = (a * shingle_hash + b) % self.prime
                if perm_hash < signature[i]:
                    signature[i] = perm_hash
        return signature

    def estimate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity between two signatures"""
        matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
        return matches / len(sig1)
