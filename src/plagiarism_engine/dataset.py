import pandas as pd
from typing import List, Dict

def load_text_pair_files(file_a: str, file_b: str) -> tuple:
    with open(file_a, 'r') as f1, open(file_b, 'r') as f2:
        text_a = f1.read()
        text_b = f2.read()
    return (text_a, text_b)

def load_corpus_directory(data_dir: str) -> Dict[str, str]:
    import os
    from glob import glob
    
    texts = {}
    for file_path in glob(f"{data_dir}/*.txt"):
        with open(file_path, 'r') as f:
            texts[file_path] = f.read()
    return texts

def read_pairs_csv(file_path: str) -> pd.DataFrame:
    required_columns = ["text_a", "text_b", "label"]
    df = pd.read_csv(file_path)
    
    if any(col not in df.columns for col in required_columns):
        raise ValueError("Missing required columns")
    
    return df
