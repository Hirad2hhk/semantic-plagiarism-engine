from collections import defaultdict
from typing import Dict, List, Set, Tuple
from typing_extensions import Dict


class LSH:
    def __init__(self, bands: int = 10,
        row_per_band: int = 8):
            self.bands = bands
            self.row_per_band = row_per_band
            self.hash_tables = Dict[int,Dict[tuple[int ,...],set[str]]]= {
                i :defaultdict(set) for i in range(bands)
            }


    def add_document(self, doc_id: str, signature: List[int]):
        """Add document signature to LSH structure"""
        expected_len = self.bands* self.row_per_band
        if len(signature) != expected_len:
                    raise ValueError(f"Signature length must be exactly {expected_len}")
        for table_i in range(self.bands):
            start = table_i * self.row_per_band
            end = start + self.row_per_band

            band_portion = tuple(signature[start:end])

            self.hash_tables[table_i][band_portion].add(doc_id)

    def get_candidate_pairs(self, threshold: float = 0.7) -> List[tuple]:
        """Retrieve candidate pairs above similarity threshold"""
        candidate_pairs = set()

        for table_idx in range(self.num_hash_tables):
            for bucket_key, doc_ids in self.hash_tables[table_idx].items():
                if len(doc_ids) > 1:
                    doc_list = list(doc_ids)
                    for i in range(len(doc_list)):
                        for j in range(i + 1, len(doc_list)):
                            pair = tuple(sorted([doc_list[i], doc_list[j]]))
                            candidate_pairs.add(pair)

        return list(candidate_pairs)

