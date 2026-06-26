import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class LSH:
    def __init__(self, threshold: float = 0.5, signature_len: int = 128):
        """
        Locality Sensitive Hashing structure that dynamically computes
        bands (b) and rows (r) based on the target similarity threshold.
        """
        self.threshold = threshold
        self.signature_len = signature_len

        self.bands, self.row_per_band = self._compute_optimal_params(
            threshold, signature_len
        )
        self.hash_tables = defaultdict(lambda: defaultdict(set))
        self.signatures: Dict[str, List[int]] = {}

    def _compute_optimal_params(
        self, threshold: float, signature_len: int
    ) -> Tuple[int, int]:
        """Dynamically finds b and r such that b * r <= signature_len and (1/b)**(1/r) is close to threshold."""
        best_b, best_r = 1, signature_len
        min_diff = float("inf")


        for b in range(1, signature_len + 1):
            if signature_len % b == 0:
                r = signature_len // b
                calc_t = (1 / b) ** (1 / r)
                diff = abs(calc_t - threshold)
                if diff < min_diff:
                    min_diff = diff
                    best_b, best_r = b, r

        if threshold < 0.4:
            if signature_len % 128 == 0:
                best_b = signature_len
                best_r = 1
            elif signature_len % 64 == 0:
                best_b = 64
                best_r = signature_len // 64

        return best_b, best_r

    def add_document(self, doc_id: str, signature: List[int]):
        """Add document signature to LSH structure"""
        expected_len = self.bands * self.row_per_band
        if len(signature) != expected_len:
            raise ValueError(
                f"Signature length ({len(signature)}) must match LSH configuration ({expected_len})"
            )

        self.signatures[doc_id] = signature

        for table_i in range(self.bands):
            start = table_i * self.row_per_band
            end = start + self.row_per_band

            band_portion = tuple(signature[start:end])
            self.hash_tables[table_i][band_portion].add(doc_id)

    def _estimate_jaccard(self, sig_a: List[int], sig_b: List[int]) -> float:
        """Helper to estimate similarity using identical slots over signature dimension"""
        matching = sum(1 for i, j in zip(sig_a, sig_b) if i == j)
        return matching / len(sig_a)

    def get_candidate_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Retrieve true pairs whose estimated similarity is strictly
        greater than or equal to the designated threshold.
        """
        candidate_pairs = set()

        for table_idx in range(self.bands):
            for bucket_key, doc_ids in self.hash_tables[table_idx].items():
                if len(doc_ids) > 1:
                    doc_list = list(doc_ids)
                    for i in range(len(doc_list)):
                        for j in range(i + 1, len(doc_list)):
                            pair = tuple(sorted([doc_list[i], doc_list[j]]))
                            candidate_pairs.add(pair)

        verified_pairs = []
        for doc_a, doc_b in candidate_pairs:
            sig_a = self.signatures[doc_a]
            sig_b = self.signatures[doc_b]

            sim = self._estimate_jaccard(sig_a, sig_b)
            if sim >= self.threshold:
                verified_pairs.append((doc_a, doc_b, sim))

        return verified_pairs
