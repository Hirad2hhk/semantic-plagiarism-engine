import hashlib
from typing import Dict, List, Optional
class SimHashEngine:
    def __init__(self, tfidf_weights: Optional[Dict[str, float]] = None):
        self.weights = tfidf_weights if tfidf_weights is not None else {}
        self.hash_bits = 64
    def _hash_token(self, token: str) -> int:
        """Helper to generate a deterministic 64-bit hash for a token"""
        hasher = hashlib.md5(token.encode('utf-8'))
        return int(hasher.hexdigest()[:16], 16)    
        
    def compute_fingerprint(self, tokens) -> int:
        """Compute SimHash fingerprint for input text"""
        v = [0.0] * self.hash_bits
                
        for token in tokens:
            weight = self.weights.get(token, 1.0)
            token_hash = self._hash_token(token)
                    
            for i in range(self.hash_bits):
                if (token_hash >> i) & 1:
                    v[i] += weight
                else:
                    v[i] -= weight
                            
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
                        
        return fingerprint
    
    def hamming_distance(self, fp1: int, 
                         fp2: int) -> float:
        """Calculate Hamming distance between two fingerprints"""
        xor_result = fp1 ^ fp2
        return bin(xor_result).count('1') / self.hash_bits
